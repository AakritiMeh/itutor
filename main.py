import components
import movie
import nltk_pipeline as summariser

results = summariser.main()


ovs = results["overall_summary"]

md = ""

for module_id, module in results["modular_summaries"].items():
    md += f"\n MODULE {module_id + 1}: {module['theme']}"
    md += f"   \nSummary: {module['summary']}"
    if module["keywords"]:
        md += f"   \nKey Terms: {', '.join(module['keywords'][:5])}"

kw = ""

for keyword, details in results["global_keywords"].items():
    kw += f"\n {keyword} (Relevance: {details['relevance_score']})"
    kw += f"\n    Context: {details['reference']}"

print(ovs)
print(md)
print(kw)


kws_dict = results["global_keywords"]

components.main(ovs, md, kw, kws_dict)

movie.main()
