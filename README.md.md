# AI Smart Cultural Storyteller



AI Smart Cultural Storyteller is an interactive AI platform that narrates folk tales, historical stories, and cultural heritage from Haryana, Punjab, and Rajasthan. It can generate audio narration, images, and videos in multiple languages, bringing stories to life with visuals, voice, and emotion.



### Features



Multilingual storytelling: Supports Hindi, Punjabi, English, Haryanvi, and other Indian dialects.



Audio narration: Generates expressive and natural-sounding text-to-speech narration.



Visual storytelling: Creates illustrations and images for story scenes using AI.



Video creation: Combines narration and visuals to produce short story videos.



Cultural preservation: Focuses on folk tales, historical events, and traditions of Haryana, Punjab, and Rajasthan.



Interactive and customizable: Choose story, language, tone (dramatic, humorous, educational), and style.



### Folder Structure



ai\_cultural\_storyteller/

│

├── data/

│   └── folk\_tales.json        # Story database

│

├── output/

│   ├── story\_audio.wav        # Generated narration

│   ├── story\_image.png        # Generated scene

│   └── story\_video.mp4        # Generated story video

│

├── storyteller.py             # Main script

├── requirements.txt           # Libraries

└── README.md                  # Project documentation

### 

### Installation



#### Clone the repository:



git clone https://github.com/yourusername/ai\_cultural\_storyteller.git

cd ai\_smart\_cultural\_storyteller



#### Install Python dependencies:



pip install -r requirements.txt



Get your Gemini API key for text/image generation:



Sign up at Google Gemini



Go to Settings → Access Tokens → Create new token



Replace GEMINI\_API\_KEY in storyteller.py



### Usage



Run the storytelling script: python storyteller.py



### Output



Audio narration: output/story\_audio.wav



Image: output/story\_image.png



Video: output/story\_video.mp4 



### Technologies Used



Language Models: Hugging Face Transformers, LLaMA, GPT



Text-to-Speech: Coqui TTS, ElevenLabs



Image Generation: Stable Diffusion, DALL·E



Video Generation: RunwayML, Pika Labs



Backend: Python (FastAPI/Flask)



Frontend (future): React.js / Streamlit



Storage: Local / Cloud (AWS S3, GCP)



### License



This project is open-source under the MIT License. You are free to modify and redistribute with attribution.

