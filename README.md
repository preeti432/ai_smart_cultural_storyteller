AI Smart Cultural Storyteller

AI Smart Cultural Storyteller is an interactive AI platform that narrates folk tales, historical stories, and cultural heritage from Haryana, Punjab, and Rajasthan. It can generate audio narration, images, and videos in multiple languages, bringing stories to life with visuals, voice, and emotion.

Features

Multilingual storytelling: Supports Hindi, Punjabi, English, Haryanvi, and other Indian dialects.

Audio narration: Generates expressive and natural-sounding text-to-speech narration.

Visual storytelling: Creates illustrations and images for story scenes using AI.

Video creation: Combines narration and visuals to produce short story videos.

Cultural preservation: Focuses on folk tales, historical events, and traditions of Haryana, Punjab, and Rajasthan.

Interactive and customizable: Choose story, language, tone (dramatic, humorous, educational), and style.

Folder Structure

ai_cultural_storyteller/
│
├── data/
│   └── folk_tales.json        # Story database
│
├── output/
│   ├── story_audio.wav        # Generated narration
│   ├── story_image.png        # Generated scene
│   └── story_video.mp4        # Generated story video
│
├── storyteller.py             # Main script
├── requirements.txt           # Libraries
└── README.md                  # Project documentation

Installation

Clone the repository:

git clone https://github.com/yourusername/ai_cultural_storyteller.git
cd ai_smart_cultural_storyteller

Install Python dependencies:

pip install -r requirements.txt

Get your Gemini API key for text/image generation:

Sign up at Google Gemini

Go to Settings → Access Tokens → Create new token

Replace GEMINI_API_KEY in storyteller.py

Usage

Run the storytelling script: python storyteller.py

Output

Audio narration: output/story_audio.wav

Image: output/story_image.png

Video: output/story_video.mp4 

Technologies Used

Language Models: Hugging Face Transformers, LLaMA, GPT

Text-to-Speech: Coqui TTS, ElevenLabs

Image Generation: Stable Diffusion, DALL·E

Video Generation: RunwayML, Pika Labs

Backend: Python (FastAPI/Flask)

Frontend (future): React.js / Streamlit

Storage: Local / Cloud (AWS S3, GCP)

License

This project is open-source under the MIT License. You are free to modify and redistribute with attribution.
