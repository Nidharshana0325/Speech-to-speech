import torch
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS  # Using Google Text-to-Speech
import os
import re
import pygame  # For audio playback
import io
import time

# ==========[ Setup Configs ]==========
model_path = "models/deepseek-1.5B-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}
).to("cpu")

# ==========[ Speech Setup ]===========
recognizer = sr.Recognizer()
pygame.mixer.init()  # Initialize pygame mixer for audio playback

# ===========[ ALL YOUR ORIGINAL FUNCTIONS UNCHANGED ]============
def classify_question_type(question):
    q = question.lower()
    if any(word in q for word in ["what is", "who is", "define", "explain"]):
        return "explanatory"
    elif any(word in q for word in ["why", "how", "describe", "compare"]):
        return "detailed"
    elif any(word in q for word in ["is", "can", "does", "did"]):
        return "direct"
    else:
        return "general"

def generate_prompt(question, qtype):
    instructions = {
        "direct": "Answer the question clearly and concisely.",
        "explanatory": "Provide a clear and detailed explanation with examples if needed.",
        "detailed": "Provide a well-structured, step-by-step explanation with a logical conclusion.",
        "general": "Respond thoughtfully with appropriate context."
    }
    return f"<think>\n{instructions[qtype]}\n\nQuestion: {question}\nAnswer:\n</think>"

def is_response_complete(text):
    return text.endswith(('.', '?', '!', '”', '’')) or len(text.split()) < 30

def ask_model(question):
    qtype = classify_question_type(question)
    prompt = generate_prompt(question, qtype)

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    max_new_tokens = 150

    while True:
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True).replace("<think>", "").replace("</think>", "").strip()

        if is_response_complete(response) or max_new_tokens >= 700:
            break

        max_new_tokens += 100

    return response

# ===========[ MODIFIED TTS FUNCTION USING gTTS ]============
def speak(text):
    # Text preprocessing
    text = re.sub(r"[\r\n]+", " ", text)
    text = text.replace("color", "colour")
    text = text.replace("favorite", "favourite")
    text = text.replace("realize", "realise")
    
    print(f"🗣️ Speaking: {text[:100]}...")  # Truncate long text in print

    try:
        # Create gTTS object with Indian English accent (en-in)
        tts = gTTS(text=text, lang='en-in', slow=False)
        
        # Save to memory file
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        
        # Play the audio
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error in speech synthesis: {e}")

# ===========[ MAIN LOOP UNCHANGED ]============
print("🎤 Speak your question. Press Ctrl+C to exit.")

while True:
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("🕑 Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=12)
            user_input = recognizer.recognize_google(audio)
            print(f"🗣️ You said: {user_input}")

            bot_output = ask_model(user_input)
            print(f"🤖 Bot: {bot_output}")

            speak(bot_output)

    except sr.UnknownValueError:
        print("⚠️ Could not understand audio. Please speak clearly.")
    except sr.WaitTimeoutError:
        print("⌛ Timeout. Try again...")
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
