import torch
import pyttsx3
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========[ Setup Configs ]==========
model_path = "deepseek-1.5B-finetuned"

# Load tokenizer and model without GPU/quantization
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}  # Force CPU
).to("cpu")

# ===========[ Speech Setup ]============
recognizer = sr.Recognizer()
tts = pyttsx3.init()

# Optional: tweak speaking rate or voice
tts.setProperty("rate", 165)

# ===========[ Question Classifier ]============
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

# ===========[ Prompt Generator ]============
def generate_prompt(question, qtype):
    if qtype == "direct":
        return f"<think>\nAnswer the question clearly and concisely.\n\nQuestion: {question}\nAnswer:\n</think>"
    elif qtype == "explanatory":
        return f"<think>\nProvide a clear and detailed explanation with examples if needed.\n\nQuestion: {question}\nExplanation:\n</think>"
    elif qtype == "detailed":
        return f"<think>\nProvide a well-structured, step-by-step explanation with a logical conclusion.\n\nQuestion: {question}\nExplanation:\n</think>"
    else:
        return f"<think>\nRespond thoughtfully with appropriate context.\n\nQuestion: {question}\nAnswer:\n</think>"

# ===========[ Check Completion ]============
def is_response_complete(text):
    return text.endswith(('.', '?', '!', '”', '’')) or len(text.split()) < 30

# ===========[ Generate Answer ]============
def ask_model(question):
    question_type = classify_question_type(question)
    prompt = generate_prompt(question, question_type)

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    max_new_tokens = 150

    while True:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        response = response.replace("<think>", "").replace("</think>", "").strip()

        if is_response_complete(response) or max_new_tokens >= 700:
            break

        max_new_tokens += 100

    return response

# ===========[ Main Loop ]============
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

            # Speak the response
            tts.say(bot_output)
            tts.runAndWait()

    except sr.UnknownValueError:
        print("⚠️ Could not understand audio. Please speak clearly.")
    except sr.WaitTimeoutError:
        print("⌛ Timeout. Try again...")
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
