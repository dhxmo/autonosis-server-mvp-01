import requests
from fastapi import HTTPException


def transcribe_audio_file(whisper_model, file_path):
    try:
        segments = whisper_model.transcribe(file_path)

        all_text = ""
        for segment in segments:
            all_text += segment.text + " "

        return all_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# def llm(pipeline, prev_diagnosis, user_prompt):
#         try:
#         messages = [
#             {
#                 "role": "system",
#                 "content": "You are a radiologists typing assitant. "
#                 "You need to edit a new text against the previous provided one. Edit the "
#                 "prev_diagnosis to accommodate what the new_diagnosis wants to add. "
#                 "Maintain the structure of prev_diagnosis, just update the information from "
#                 "new_diagnosis into prev_diagnosis. Provide the result in the exact same "
#                 "format as the prev_diagnosis. If markdown characters \n \t are present return them as is."
#                 "There might be transcription errors in new_diagnosis due to misunderstanding. "
#                 "If words don't align with the context of radiology, edit them for what you see fit. "
#                 "Remember: provide the result in the exact same format as the prev_diagnosis.",
#             },
#             {
#                 "role": "user",
#                 "content": f"prev_diagnosis: {prev_diagnosis} \n\n new_diagnosis: {user_prompt}",
#             },
#         ]
#
#         outputs = pipeline(messages, max_new_tokens=128)
#         response = outputs[0]["generated_text"][-1]
#
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


def ollama_llm(prev_diagnosis, user_prompt):
    url = "http://localhost:11434/api/chat"
    data = {
        "model": "phi4:latest",
        "messages": [
            {
                "role": "system",
                "content": "You are a radiologists typing assistant. "
                "You need to edit a new text against the previous provided one. Edit the "
                "prev_diagnosis to accommodate what the new_diagnosis wants to add. "
                "Maintain the structure of prev_diagnosis, just update the information from "
                "new_diagnosis into prev_diagnosis. Provide the result in the exact same "
                "format as the prev_diagnosis. If markdown characters \\n \\t are present return them as is."
                "There might be transcription errors in new_diagnosis due to misunderstanding. "
                "If words don't align with the context of radiology, edit them for what you see fit in context with the rest of the new_diagnosis."
                "Remember: provide the result in the exact same format as the prev_diagnosis.",
            },
            {
                "role": "user",
                "content": f"prev_diagnosis: {prev_diagnosis} \n\n new_diagnosis: {user_prompt}",
            },
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        print("response.json()", response.json(), response.json()["message"]["content"])

        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
