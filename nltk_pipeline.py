import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

import json
import re
import warnings
from collections import Counter, defaultdict

import nltk
import numpy as np
import PyPDF2
import torch
from keybert import KeyBERT
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")
output_path = ""

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class FlexibleSemanticProcessor:
    def __init__(self):

        print("Loading models...")

        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1,
        )

        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.kw_model = KeyBERT(model=self.sentence_model)

        print("Models loaded successfully!")

    def extract_text_from_pdf(self, pdf_path):

        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return None

    def preprocess_text(self, text):

        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\]", "", text)

        text = re.sub(r"(?:Figure|Fig\.|Table|Tab\.)\s*\d+[a-zA-Z]*", "", text)

        text = re.sub(r"\bPage\s*\d+\b", "", text)
        text = re.sub(r"\b\d+\s*Page\b", "", text)

        return text.strip()

    def detect_natural_breakpoints(self, text):

        sentences = sent_tokenize(text)
        breakpoints = []

        embeddings = self.sentence_model.encode(sentences)

        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(similarity)

        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            threshold = mean_sim - 1.2 * std_sim

            for i, sim in enumerate(similarities):
                if sim < threshold and i > 2:
                    breakpoints.append(i + 1)

        for i, sentence in enumerate(sentences):
            sentence_clean = sentence.strip()

            if sentence_clean.endswith("?") and len(sentence_clean.split()) > 8:
                breakpoints.append(i + 1)

            if i > 0 and i < len(sentences) - 1:
                current_len = len(sentence_clean.split())
                prev_len = len(sentences[i - 1].split())
                next_len = len(sentences[i + 1].split())

                if current_len > 1.8 * prev_len and current_len > 1.8 * next_len:
                    breakpoints.append(i + 1)

        breakpoints = sorted(list(set(breakpoints)))

        filtered_breakpoints = []
        for bp in breakpoints:
            if not filtered_breakpoints or bp - filtered_breakpoints[-1] > 3:
                filtered_breakpoints.append(bp)

        return filtered_breakpoints

    def create_semantic_segments(self, sentences, method="hybrid"):

        if len(sentences) < 4:
            return {0: sentences}

        embeddings = self.sentence_model.encode(sentences)

        segments = {}

        if method == "clustering":
            segments = self._cluster_based_segmentation(sentences, embeddings)
        elif method == "topic_modeling":
            segments = self._topic_modeling_segmentation(sentences)
        elif method == "sliding_window":
            segments = self._sliding_window_segmentation(sentences, embeddings)
        else:
            segments = self._hybrid_segmentation(sentences, embeddings)

        return segments

    def _cluster_based_segmentation(self, sentences, embeddings):

        from sklearn.metrics import silhouette_score

        max_clusters = min(8, len(sentences) // 3)
        best_n_clusters = 2
        best_score = -1

        if max_clusters > 2:
            for n_clusters in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                except:
                    continue

        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[int(label)].append((i, sentences[i]))
        sorted_clusters = {}
        for cluster_id, sent_list in clusters.items():
            sent_list.sort(key=lambda x: x[0])
            sorted_clusters[int(cluster_id)] = [sent[1] for sent in sent_list]

        return sorted_clusters

    def _topic_modeling_segmentation(self, sentences):

        vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english", ngram_range=(1, 2)
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except:

            return self._simple_sequential_segmentation(sentences)

        n_topics = min(6, max(2, len(sentences) // 5))

        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=100
        )
        lda.fit(tfidf_matrix)

        topic_distributions = lda.transform(tfidf_matrix)
        topic_assignments = np.argmax(topic_distributions, axis=1)

        topics = defaultdict(list)
        for i, topic in enumerate(topic_assignments):
            topics[int(topic)].append((i, sentences[i]))
        sorted_topics = {}
        for topic_id, sent_list in topics.items():
            sent_list.sort(key=lambda x: x[0])
            sorted_topics[int(topic_id)] = [sent[1] for sent in sent_list]

        return sorted_topics

    def _sliding_window_segmentation(self, sentences, embeddings):

        segments = defaultdict(list)
        current_segment = 0
        window_size = min(3, len(sentences) // 4)

        if window_size < 1:
            return {0: sentences}

        for i in range(len(sentences) - window_size):
            if i + 2 * window_size <= len(embeddings):
                window1 = embeddings[i : i + window_size]
                window2 = embeddings[i + window_size : i + 2 * window_size]

                avg_emb1 = np.mean(window1, axis=0)
                avg_emb2 = np.mean(window2, axis=0)

                similarity = cosine_similarity([avg_emb1], [avg_emb2])[0][0]

                if similarity < 0.6 and len(segments[current_segment]) > 2:
                    current_segment += 1

            segments[current_segment].append(sentences[i])

        for i in range(max(0, len(sentences) - window_size), len(sentences)):
            if sentences[i] not in segments[current_segment]:
                segments[current_segment].append(sentences[i])

        return {int(k): v for k, v in segments.items()}

    def _hybrid_segmentation(self, sentences, embeddings):

        cluster_segments = self._cluster_based_segmentation(sentences, embeddings)
        topic_segments = self._topic_modeling_segmentation(sentences)

        breakpoints = self.detect_natural_breakpoints(" ".join(sentences))

        if breakpoints and len(breakpoints) > 0:
            segments = {}
            segment_id = 0
            current_start = 0

            for bp in breakpoints + [len(sentences)]:
                if bp > current_start:
                    segments[int(segment_id)] = sentences[current_start:bp]
                    segment_id += 1
                    current_start = bp

            return segments
        else:

            return cluster_segments

    def _simple_sequential_segmentation(self, sentences):

        segment_size = max(3, len(sentences) // 4)
        segments = {}

        for i in range(0, len(sentences), segment_size):
            segment_id = i // segment_size
            segments[int(segment_id)] = sentences[i : i + segment_size]

        return segments

    def identify_segment_themes_dynamically(self, segments):

        themed_segments = {}

        for segment_id, sentences in segments.items():
            segment_text = " ".join(sentences)

            theme, keywords = self._extract_meaningful_theme_and_keywords(segment_text)

            themed_segments[segment_id] = {
                "theme": theme,
                "sentences": sentences,
                "keywords": keywords,
                "coherence_score": self._calculate_segment_coherence(sentences),
            }

        return themed_segments

    def _extract_meaningful_theme_and_keywords(self, text):

        try:

            cleaned_text = self._deep_clean_text(text)

            if len(cleaned_text.strip()) > 20:
                keywords = self.kw_model.extract_keywords(
                    cleaned_text,
                    keyphrase_ngram_range=(2, 4),
                    stop_words="english",
                    top_n=8,
                    use_mmr=True,
                    diversity=0.8,
                    nr_candidates=50,
                )

                valid_keywords = []
                for kw, score in keywords:
                    kw_clean = kw.strip().lower()

                    if (
                        len(kw_clean) >= 4
                        and not kw_clean.isdigit()
                        and len(kw_clean.split()) >= 1
                        and all(len(word) >= 2 for word in kw_clean.split())
                        and any(char.isalpha() for char in kw_clean)
                        and score > 0.1
                    ):
                        valid_keywords.append((kw.title(), score))

                if len(valid_keywords) >= 2:
                    theme = self._create_theme_from_keywords(valid_keywords)
                    keyword_list = [kw for kw, _ in valid_keywords[:5]]
                    return theme, keyword_list
        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")

        try:
            theme, keywords = self._tfidf_based_extraction(text)
            if theme != "General Content":
                return theme, keywords
        except Exception as e:
            print(f"TF-IDF extraction failed: {e}")

        try:
            theme, keywords = self._advanced_frequency_analysis(text)
            return theme, keywords
        except Exception as e:
            print(f"Frequency analysis failed: {e}")

        return "General Content", ["content", "information", "text"]

    def _deep_clean_text(self, text):

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:]", " ", text)

        words = text.split()
        cleaned_words = []
        for word in words:
            word = word.strip(".,!?;:")
            if len(word) >= 2 and word.isalpha():
                cleaned_words.append(word)

        return " ".join(cleaned_words)

    def _tfidf_based_extraction(self, text):

        from sklearn.feature_extraction.text import TfidfVectorizer

        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return "General Content", ["content"]

        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.8,
            token_pattern=r"\b[a-zA-Z]{3,}\b",
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()

            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            top_indices = mean_scores.argsort()[-10:][::-1]
            top_keywords = [
                feature_names[i] for i in top_indices if mean_scores[i] > 0.1
            ]

            if len(top_keywords) >= 2:

                theme_words = top_keywords[:3]
                theme = " & ".join([word.title() for word in theme_words[:2]])
                return theme, top_keywords[:5]

        except Exception:
            pass

        return "General Content", ["content"]

    def _advanced_frequency_analysis(self, text):

        import re
        from collections import Counter

        words = word_tokenize(text.lower())

        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "that",
            "this",
            "these",
            "those",
            "also",
            "however",
            "therefore",
            "thus",
            "furthermore",
            "moreover",
            "their",
            "there",
            "they",
            "them",
            "then",
            "than",
            "when",
            "where",
            "who",
            "what",
            "which",
            "how",
            "why",
        }

        meaningful_words = []
        for word in words:
            if (
                len(word) >= 4
                and word.isalpha()
                and word not in stop_words
                and not word.isdigit()
            ):
                meaningful_words.append(word)

        word_freq = Counter(meaningful_words)

        tech_words = []
        for word in meaningful_words:
            if len(word) >= 6 and any(
                substr in word
                for substr in [
                    "tion",
                    "ing",
                    "ment",
                    "ity",
                    "ness",
                    "ical",
                    "ogy",
                    "ism",
                ]
            ):
                tech_words.append(word)

        all_candidates = list(word_freq.keys()) + tech_words
        final_candidates = Counter(all_candidates)

        top_words = [
            word
            for word, count in final_candidates.most_common(8)
            if count >= 1 and len(word) >= 4
        ]

        if len(top_words) >= 2:

            theme = f"{top_words[0].title()} & {top_words[1].title()}"
            return theme, top_words[:5]
        elif len(top_words) == 1:
            return top_words[0].title(), top_words

        original_words = re.findall(r"\b[A-Z][a-z]{3,}\b", text)
        if original_words:
            unique_words = list(set(original_words))[:5]
            if len(unique_words) >= 2:
                theme = f"{unique_words[0]} & {unique_words[1]}"
                return theme, unique_words

        return "General Content", ["content", "information"]

    def _create_theme_from_keywords(self, valid_keywords):
        """Create a meaningful theme from valid keywords"""
        if len(valid_keywords) >= 2:

            kw1 = valid_keywords[0][0]
            kw2 = valid_keywords[1][0]

            if len(kw1) > 15:
                kw1 = kw1.split()[0]
            if len(kw2) > 15:
                kw2 = kw2.split()[0]

            theme = f"{kw1} & {kw2}"
            return theme
        elif len(valid_keywords) == 1:
            return valid_keywords[0][0]
        else:
            return "General Content"

    def _calculate_segment_coherence(self, sentences):

        if len(sentences) < 2:
            return 1.0

        try:
            embeddings = self.sentence_model.encode(sentences)
            similarities = []

            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(float(sim))  # Convert to Python float

            return float(np.mean(similarities)) if similarities else 0.5
        except:
            return 0.5

    def generate_adaptive_summaries(self, themed_segments):

        modular_summaries = {}

        for segment_id, segment_info in themed_segments.items():
            segment_text = " ".join(segment_info["sentences"])

            if len(segment_text.split()) < 25:

                summary = segment_text
            else:
                try:

                    original_length = len(segment_text.split())
                    coherence = segment_info["coherence_score"]
                    base_ratio = 0.3 if coherence > 0.7 else 0.4

                    max_length = min(200, max(30, int(original_length * base_ratio)))
                    min_length = max(15, max_length // 3)

                    summary_result = self.summarizer(
                        segment_text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                    )
                    summary = summary_result[0]["summary_text"]

                except Exception as e:
                    print(f"Error summarizing segment {segment_id}: {e}")

                    sentences = sent_tokenize(segment_text)
                    summary = " ".join(sentences[: min(3, len(sentences))])

            modular_summaries[segment_id] = {
                "theme": segment_info["theme"],
                "summary": summary,
                "keywords": segment_info["keywords"],
                "coherence_score": float(segment_info["coherence_score"]),
                "original_length": len(segment_text),
                "summary_length": len(summary),
                "compression_ratio": (
                    float(len(summary) / len(segment_text))
                    if len(segment_text) > 0
                    else 1.0
                ),
            }

        return modular_summaries

    def create_overall_summary(self, modular_summaries):

        try:

            combined_summaries = []
            for segment in modular_summaries.values():
                combined_summaries.append(segment["summary"])

            combined_text = " ".join(combined_summaries)

            if len(combined_text.split()) > 1220:
                overall_summary = self.summarizer(
                    combined_text, max_length=1500, min_length=50, do_sample=False
                )
                return overall_summary[0]["summary_text"]
            else:
                return combined_text

        except Exception as e:
            print(f"Error creating overall summary: {e}")

            return "Overall summary could not be generated."

    def extract_global_keywords(self, full_text, top_n=15):

        try:

            _, global_keywords_list = self._extract_meaningful_theme_and_keywords(
                full_text
            )

            if len(global_keywords_list) < top_n:

                try:
                    _, additional_keywords = self._tfidf_based_extraction(full_text)
                    global_keywords_list.extend(additional_keywords)
                except:
                    pass

            seen = set()
            unique_keywords = []
            for kw in global_keywords_list:
                if kw.lower() not in seen:
                    unique_keywords.append(kw)
                    seen.add(kw.lower())

            keyword_dict = {}
            sentences = sent_tokenize(full_text)

            for keyword in unique_keywords[:top_n]:
                reference = self._find_best_reference_sentence(keyword, sentences)

                word_count = full_text.lower().count(keyword.lower())
                relevance_score = min(1.0, (word_count * len(keyword)) / 100)

                keyword_dict[keyword] = {
                    "relevance_score": float(round(relevance_score, 3)),
                    "reference": reference,
                }

            return keyword_dict

        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return {
                "content": {
                    "relevance_score": 0.5,
                    "reference": "No reference available",
                }
            }

    def _find_best_reference_sentence(self, keyword, sentences):

        best_sentence = None
        best_match_score = 0

        keyword_lower = keyword.lower()
        keyword_words = set(word.lower() for word in keyword.split() if len(word) > 2)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(
                word.lower() for word in word_tokenize(sentence) if word.isalpha()
            )

            match_score = 0

            if keyword_lower in sentence_lower:
                match_score = 1.0

            elif keyword_words and sentence_words:
                overlap = len(keyword_words.intersection(sentence_words))
                match_score = overlap / len(keyword_words)

            if match_score > best_match_score:
                best_match_score = match_score
                best_sentence = sentence

        if best_sentence:
            return (
                best_sentence[:200] + "..."
                if len(best_sentence) > 200
                else best_sentence
            )
        else:
            return "Reference not found"

    def process_document(
        self, input_path, output_path=None, is_pdf=True, segmentation_method="hybrid"
    ):

        print("Starting flexible document processing...")

        # Extract text
        if is_pdf:
            text = self.extract_text_from_pdf(input_path)
        else:
            with open(input_path, "r", encoding="utf-8") as file:
                text = file.read()

        if not text:
            return {"error": "Could not extract text from document"}

        print("Text extracted successfully!")

        cleaned_text = self.preprocess_text(text)
        print(f"Text preprocessed. Length: {len(cleaned_text)} characters")

        print("Creating semantic segments...")
        sentences = sent_tokenize(cleaned_text)
        # MAX_SENTENCE_WORDS = 200
        # processed_sentences = []
        # for sentence in sentences:
        #     words = sentence.split()
        #     if len(words) > MAX_SENTENCE_WORDS:
        #         # If a sentence is too long, split it into smaller chunks
        #         for i in range(0, len(words), MAX_SENTENCE_WORDS):
        #             chunk = " ".join(words[i:i + MAX_SENTENCE_WORDS])
        #             processed_sentences.append(chunk)
        #     else:
        #         processed_sentences.append(sentence)
        segments = self.create_semantic_segments(sentences, method=segmentation_method)

        print(f"Created {len(segments)} segments using {segmentation_method} method")

        print("Identifying segment themes...")
        themed_segments = self.identify_segment_themes_dynamically(segments)

        print("Generating adaptive summaries...")
        modular_summaries = self.generate_adaptive_summaries(themed_segments)

        print("Creating overall summary...")
        overall_summary = self.create_overall_summary(modular_summaries)

        print("Extracting global keywords...")
        global_keywords = self.extract_global_keywords(cleaned_text)

        avg_coherence = np.mean(
            [seg["coherence_score"] for seg in themed_segments.values()]
        )
        avg_compression = np.mean(
            [mod["compression_ratio"] for mod in modular_summaries.values()]
        )

        def convert_numpy_types(obj):
            if isinstance(obj, dict):

                converted_dict = {}
                for key, value in obj.items():

                    if isinstance(key, (np.integer, np.int32, np.int64)):
                        new_key = int(key)
                    elif isinstance(key, (np.floating, np.float32, np.float64)):
                        new_key = float(key)
                    else:
                        new_key = key
                    converted_dict[new_key] = convert_numpy_types(value)
                return converted_dict
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        results = {
            "overall_summary": overall_summary,
            "modular_summaries": convert_numpy_types(modular_summaries),
            "global_keywords": convert_numpy_types(global_keywords),
            "processing_info": {
                "segmentation_method": segmentation_method,
                "num_original_sentences": len(sentences),
                "num_segments": len(segments),
                "average_coherence": float(round(avg_coherence, 3)),
                "average_compression_ratio": float(round(avg_compression, 3)),
            },
            "document_stats": {
                "original_length": len(text),
                "processed_length": len(cleaned_text),
                "total_keywords": len(global_keywords),
            },
        }

        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(results, file, indent=2, ensure_ascii=False)
                print(f"Results saved to {output_path}")
            except Exception as e:
                print(f"Error saving results: {e}")

        try:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_output_path = (
                f"detailed_results_{segmentation_method}_{timestamp}.json"
            )

            detailed_results = {
                "metadata": {
                    "timestamp": timestamp,
                    "segmentation_method": segmentation_method,
                    "document_source": input_path,
                    "processing_version": "2.0",
                },
                "results": results,
            }

            with open(detailed_output_path, "w", encoding="utf-8") as file:
                json.dump(detailed_results, file, indent=2, ensure_ascii=False)
            print(f"Detailed results saved to {detailed_output_path}")
        except Exception as e:
            print(f"Error saving detailed results: {e}")

        return results, output_path

    def format_output(self, results):

        if "error" in results:
            return results

        print("\n" + "=" * 100)
        print("FLEXIBLE SEMANTIC ANALYSIS RESULTS")
        print("=" * 100)

        processing_info = results["processing_info"]
        print(f"Segmentation Method: {processing_info['segmentation_method']}")
        print(f"Original Sentences: {processing_info['num_original_sentences']}")
        print(f"Segments Created: {processing_info['num_segments']}")
        print(f"Average Coherence Score: {processing_info['average_coherence']}")
        print(
            f"Average Compression Ratio: {processing_info['average_compression_ratio']}"
        )

        print("\n" + "=" * 100)
        print("OVERALL SUMMARY")
        print("=" * 100)
        print(results["overall_summary"])

        print("\n" + "=" * 100)
        print("SEMANTIC MODULES")
        print("=" * 100)

        for module_id, module in results["modular_summaries"].items():
            print(f"\n📍 MODULE {module_id + 1}: {module['theme']}")
            print(f"   Coherence Score: {module['coherence_score']:.3f}")
            print(f"   Summary: {module['summary']}")
            if module["keywords"]:
                print(f"   Key Terms: {', '.join(module['keywords'][:5])}")
            print(
                f"   Compression: {module['original_length']} → {module['summary_length']} words ({module['compression_ratio']:.2f})"
            )
            print("-" * 80)

        print("\n" + "=" * 100)
        print("GLOBAL KEYWORDS & CONCEPTS")
        print("=" * 100)

        for keyword, details in results["global_keywords"].items():
            print(f"\n🔑 {keyword} (Relevance: {details['relevance_score']})")
            print(f"    Context: {details['reference']}")

        return results


def main():
    processor = FlexibleSemanticProcessor()

    print("Processing sample quantum computing text...")

    with open("sample.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()

    methods = ["hybrid", "clustering", "topic_modeling", "sliding_window"]

    for method in methods:
        if method == "hybrid":
            print(f"\n{'='*120}")
            print(f"PROCESSING WITH {method.upper()} METHOD")
            print(f"{'='*120}")

            results, file = processor.process_document(
                "sample.txt",
                f"results_{method}.json",
                is_pdf=False,
                segmentation_method=method,
            )
            # print(results)
            # processor.format_output(results)
    return results
    # import os
    # os.remove("quantum_sample.txt")
    # for method in methods:
    #     try:
    #         os.remove(f"results_{method}.json")
    #     except:
    #         pass


if __name__ == "__main__":
    main()
