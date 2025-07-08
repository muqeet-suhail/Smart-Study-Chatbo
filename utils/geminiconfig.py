# gemini_chat.py
from utils.libraries import genai, os, load_dotenv

# Load environment variables
load_dotenv()

def get_api_key(file_path="key.txt"):
    # Check if key file exists and read it
    print("🔍 Checking for existing API key...")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            key = f.read().strip()
            if key:
                print("🔐 API key loaded from file.", key)
                return key
            else:
                print("⚠️ Key.txt file is empty.")
    else:
        print("⚠️ Key file not found.")

    # Prompt user for API key if not found
    key = input("Enter your Gemini API Key: ").strip()
    with open(file_path, "w") as f:
        f.write(key)
    print(f"✅ API key saved to {file_path}")
    return key

# Configure Gemini with key
def configure_gemini():
    api_key = get_api_key()
    genai.configure(api_key=api_key)

# Get best available model name
def get_best_available_model():
    configure_gemini()  # Ensure API key is loaded and configured

    preferred_models = [
        "models/gemini-1.5-pro",
        "models/gemini-pro",
        "models/gemini-1.5-flash",
        "models/gemini-1.0-pro",
        "models/text-bison-001"
    ]

    try:
        # List all models and filter for those supporting generateContent
        models = genai.list_models()
        usable_models = [
            model for model in models
            if "generateContent" in model.supported_generation_methods
        ]

        for preferred in preferred_models:
            if any(model.name == preferred for model in usable_models):
                print(f"🔍 Trying model: {preferred}")

                try:
                    # Test the model with a tiny prompt to see if quota allows usage
                    test_model = genai.GenerativeModel(preferred)
                    _ = test_model.generate_content("ping")  # small prompt
                    print(f"✅ Using model: {preferred}")
                    return preferred

                except Exception as inner_error:
                    if "quota" in str(inner_error).lower() or "429" in str(inner_error):
                        print(f"⚠️ Quota exceeded for {preferred}, trying next...")
                        continue  # try next model
                    else:
                        print(f"❌ Error with model {preferred}: {inner_error}")
                        continue

        raise Exception("❌ All preferred models failed due to quota or other issues.")

    except Exception as e:
        print("❌ Fatal error while listing or testing models:", e)
        return None


# Load model object
def load_model():
    model_name = get_best_available_model()
    if not model_name:
        return None
    return genai.GenerativeModel(model_name)

# Ask Gemini
def ask_gemini(model, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text
