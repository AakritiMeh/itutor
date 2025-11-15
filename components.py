import base64
import json
import os

import requests

# Get API key from environment
KEY = ""


# 1. Generate lecture script and video timeline using Gemini Pro
def generate_lecture_content(summary, modular_details, keywords):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    prompt = f"""
    Based on the following information,generate:
    
    1. A complete, engaging lecture script for a video (approximately 3-5 minutes when narrated)
    2. A JSON video timeline with timestamps and image prompts for each keyword
    
    OVERALL SUMMARY:
    {summary}
    
    MODULAR DETAILS:
    {modular_details}
    
    KEYWORDS:
    {keywords}
    
    Please return your response in this exact JSON format:
    {{
        "lecture_script": "Your complete lecture script here...",
        "video_timeline": [
            {{
                "start_time": 0,
                "end_time": 15,
                "keyword": "Music Arijit Singh",
                "image_prompt": "Young Arijit Singh learning music in Murshidabad",
                "transition": "fade"
            }},
            ...
        ]
    }}
    
    Make sure the timeline covers the entire lecture duration with appropriate transitions.
    """

    headers = {"x-goog-api-key": KEY, "Content-Type": "application/json"}

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        # Extract the text from the response
        text_response = result["candidates"][0]["content"]["parts"][0]["text"]

        # Parse JSON from the response (remove markdown code blocks if present)
        text_response = text_response.strip()
        if text_response.startswith("```"):
            text_response = text_response[7:]
        if text_response.startswith("```"):
            text_response = text_response[3:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]

        return json.loads(text_response.strip())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


# 2. Generate images for each keyword using Gemini Image API
def generate_image_for_keyword(keyword, context, image_prompt, output_path):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"

    headers = {"x-goog-api-key": KEY, "Content-Type": "application/json"}

    # Create detailed prompt
    full_prompt = f"{image_prompt}. Context: {context}. Style: Professional documentary photograph"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"responseModalities": ["Text", "image"]},
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        # Extract base64 image data
        try:
            # candidates is a LIST, so use [0]
            parts = result["candidates"][0]["content"]["parts"]

            # Find the part with inlineData (image is usually in second part)
            image_data = None
            for part in parts:
                if "inlineData" in part:
                    image_data = part["inlineData"]["data"]
                    break

            if image_data:
                # Decode and save image
                image_bytes = base64.b64decode(image_data)
                with open(output_path, "wb") as f:
                    f.write(image_bytes)

                print(f"✅ Image saved: {output_path}")
                return output_path
            else:
                print(f"❌ No image data found in response")
                print("Response structure:", json.dumps(result, indent=2))
                return None

        except (KeyError, IndexError) as e:
            print(f"❌ Error extracting image data: {e}")
            print("Full response:", json.dumps(result, indent=2))
            return None
    else:
        print(f"❌ Error generating image: {response.status_code}")
        print(response.text)
        return None


def generate_lecture_audio(lecture_script):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
    headers = {"x-goog-api-key": KEY, "Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": lecture_script}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Autonoe"}}
            },
        },
        "model": "gemini-2.5-flash-preview-tts",
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    audio_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]

    # Write .pcm file
    with open("out.pcm", "wb") as wb:
        wb.write(base64.b64decode(audio_b64))

    # Convert to wav using ffmpeg
    os.system("ffmpeg -f s16le -ar 24000 -ac 1 -i out.pcm out.wav")


# Main execution
def main(ovs, md, kw, kws_dict):
    print("Generating lecture script and timeline...")
    content = generate_lecture_content(ovs, md, kw)

    if content:
        lecture_script = content["lecture_script"]
        video_timeline = content["video_timeline"]

        # Save lecture script
        with open("lecture_script.txt", "w", encoding="utf-8") as f:
            f.write(lecture_script)
        print("\nLecture script saved to lecture_script.txt\n")

        # Save timeline
        with open("video_timeline.json", "w", encoding="utf-8") as f:
            json.dump(video_timeline, f, indent=2)
        print("Video timeline saved to video_timeline.json")

        # Generate images for timeline
        print("\nGenerating images for video timeline...")
        os.makedirs("images", exist_ok=True)

        for idx, segment in enumerate(video_timeline):
            keyword = segment.get("keyword", f"segment_{idx}")
            image_prompt = segment["image_prompt"]

            # Find context from kws_dict
            context = ""
            for kw_key, details in kws_dict.items():
                if kw_key.lower() in keyword.lower():
                    context = details["reference"]
                    break

            output_path = f"images/image_{idx:03d}_{keyword.replace(' ', '_')[:30]}.png"
            generate_image_for_keyword(keyword, context, image_prompt, output_path)

            # Add image path to timeline
            video_timeline[idx]["image_path"] = output_path

        generate_lecture_audio(lecture_script)
        print("\nLecture audio generated")
        # Save updated timeline with image paths
        with open("video_timeline_with_images.json", "w", encoding="utf-8") as f:
            json.dump(video_timeline, f, indent=2)
        print("\nUpdated timeline saved to video_timeline_with_images.json")

        print("\n✅ All content generated successfully!")
        print(f"   - Lecture script: lecture_script.txt")
        print(f"   - Timeline: video_timeline_with_images.json")
        print(f"   - Images: images/ folder")
    else:
        print("Failed to generate content")


if __name__ == "__main__":
    main()
