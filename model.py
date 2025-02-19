from pywhispercpp.model import Model


def transcribe_audio_file(file_path):
    model = Model("base.en")
    segments = model.transcribe(file_path)

    all_text = ""
    for segment in segments:
        all_text += segment.text + " "

    return all_text


def llm(pipeline, prev_diagnosis, user_prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a radiologists typing assitant. "
            "You need to edit a new text against the previous provided one. Edit the "
            "prev_diagnosis to accommodate what the new_diagnosis wants to add. "
            "Maintain the structure of prev_diagnosis, just update the information from "
            "new_diagnosis into prev_diagnosis. Provide the result in the exact same "
            "format as the prev_diagnosis. If markdown characters \n \t are present return them as is."
            "There might be transcription errors in new_diagnosis due to misunderstanding. "
            "If words don't align with the context of radiology, edit them for what you see fit. "
            "Remember: provide the result in the exact same format as the prev_diagnosis.",
        },
        {
            "role": "user",
            "content": f"prev_diagnosis: {prev_diagnosis} \n\n new_diagnosis: {user_prompt}",
        },
    ]

    outputs = pipeline(messages, max_new_tokens=128)
    response = outputs[0]["generated_text"][-1]

    return response
