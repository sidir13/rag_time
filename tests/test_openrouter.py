"""Test rapide de la clef OPENROUTER_API_KEY.

Vérifie que :
- La variable d'environnement est définie
- L'appel à l'API OpenRouter fonctionne (modèle gpt-4o)

Lancement : python test_openrouter.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("❌ OPENROUTER_API_KEY non trouvée dans l'environnement / .env")
        sys.exit(1)

    print(f"✅ Clef trouvée : {api_key[:8]}{'*' * (len(api_key) - 8)}")
    print("   Envoi d'un message test à OpenRouter...")

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        model = "openai/gpt-4o"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Réponds juste 'OK' en un mot."}],
            max_tokens=10,
        )

        answer = response.choices[0].message.content.strip()
        print(f"✅ Réponse du modèle ({model}) : {answer}")
        print("✅ La clef OpenRouter fonctionne correctement.")

    except Exception as e:
        print(f"❌ Erreur lors de l'appel API : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
