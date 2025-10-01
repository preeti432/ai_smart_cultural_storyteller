import gradio as gr
import google.generativeai as genai
import requests
import json
import base64
import os
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    import moviepy.editor as mp
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: MoviePy not installed. Video generation will be disabled.")


from gtts import gTTS
import random
import time
from datetime import datetime
import io
import urllib.parse
import asyncio
import concurrent.futures
import threading
import socket
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure API keys (you'll need to set these)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure requests session with better timeout and retry settings
session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(max_retries=2))
session.mount('https://', requests.adapters.HTTPAdapter(max_retries=2))

class CulturalStoryteller:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Heritage database for the three states
        self.heritage_data = {
            "haryana": {
                "folk_tales": [
                    "The Legend of Prithviraj Chauhan and Princess Sanyogita",
                    "The Tale of Raja Rasalu of Sialkot",
                    "The Story of Birbal and the Farmer's Wisdom",
                    "The Legend of Kurukshetra and the Mahabharata",
                    "The Tale of the Brave Jat Warriors"
                ],
                "historical_sites": [
                    "Rakhigarhi - Indus Valley Civilization's largest site",
                    "Kurukshetra - Land of the Bhagavad Gita",
                    "Panipat - Historic battlefield",
                    "Thanesar - Ancient religious center"
                ],
                "cultural_elements": [
                    "Traditional wrestling (Kushti)",
                    "Folk dances like Ghoomar",
                    "Agricultural festivals",
                    "Cattle rearing traditions"
                ],
                "visual_elements": [
                    "rural landscapes", "wrestling arenas", "agricultural fields", 
                    "traditional Haryanvi attire", "folk dance performances"
                ]
            },
            "punjab": {
                "folk_tales": [
                    "The Legend of Heer Ranjha",
                    "The Tale of Sassi Punnu",
                    "The Story of Sohni Mahiwal",
                    "The Legend of Mirza Sahiban",
                    "The Tale of Guru Nanak's Travels"
                ],
                "historical_sites": [
                    "Golden Temple - Spiritual center of Sikhism",
                    "Anandpur Sahib - Birthplace of Khalsa",
                    "Takht Sri Patna Sahib",
                    "Jallianwala Bagh - Memorial of sacrifice"
                ],
                "cultural_elements": [
                    "Bhangra and Giddha dances",
                    "Punjabi literature and poetry",
                    "Agricultural prosperity",
                    "Sikh traditions and values"
                ],
                "visual_elements": [
                    "golden temple", "vibrant festivals", "bhangra dancers", 
                    "agricultural fields", "colorful Punjabi attire"
                ]
            },
            "rajasthan": {
                "folk_tales": [
                    "The Legend of Meera Bai",
                    "The Tale of Padmavati and Alauddin Khilji",
                    "The Story of Maharana Pratap",
                    "The Legend of Dhola Maru",
                    "The Tale of the Desert Rose"
                ],
                "historical_sites": [
                    "Chittorgarh Fort - Symbol of Rajput valor",
                    "Amber Fort - Architectural marvel",
                    "Jaisalmer - The Golden City",
                    "Mehrangarh Fort - Jodhpur's crown"
                ],
                "cultural_elements": [
                    "Rajasthani folk music and dance",
                    "Desert culture and lifestyle",
                    "Rajput traditions and honor",
                    "Colorful festivals and crafts"
                ],
                "visual_elements": [
                    "desert landscapes", "forts and palaces", "camels", 
                    "traditional Rajasthani attire", "folk musicians"
                ]
            }
        }
        
        # Language support
        self.languages = {
            "english": "en",
            "hindi": "hi",
            "punjabi": "pa",
            "gujarati": "gu",
            "marathi": "mr",
            "bengali": "bn"
        }
        
        # Thread pool for parallel image generation
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)  # Reduced to 3 for stability

    def generate_story_prompt(self, state, story_type, custom_prompt=None):
        """Generate appropriate prompts for different story types"""
        base_context = f"""
        You are an expert storyteller specializing in the rich cultural heritage of {state.title()}, India. 
        Your stories should be authentic, culturally accurate, and engaging for modern audiences while 
        preserving traditional wisdom and values.
        
        IMPORTANT: Keep the story concise (300-500 words) to ensure faster generation.
        """
        
        if custom_prompt:
            return base_context + f"\n\nUser Request: {custom_prompt}\n\nCreate a captivating story (300-500 words) that incorporates elements of {state}'s culture, traditions, and heritage."
        
        elif story_type == "folk_tale":
            folk_tales = self.heritage_data[state]["folk_tales"]
            selected_tale = random.choice(folk_tales)
            return base_context + f"""
            Create a detailed retelling of the folk tale: "{selected_tale}"
            
            The story should be 300-500 words and include:
            - Rich cultural context of {state.title()}
            - Traditional values and moral lessons
            - Vivid descriptions of landscapes, customs, and traditions
            - Character development that reflects the spirit of the people
            - A narrative structure that honors oral storytelling traditions
            
            Make it engaging for modern audiences while staying true to cultural authenticity.
            """
        
        elif story_type == "historical":
            if state == "haryana":
                return base_context + """
                Create a fascinating historical narrative about Rakhigarhi, the largest Indus Valley Civilization 
                site in India, located in Haryana. The story should be 300-500 words and include:
                
                - The daily life of people in the ancient Indus Valley civilization
                - Archaeological discoveries and their significance
                - The advanced urban planning, drainage systems, and craftsmanship
                - Trade relationships with other ancient civilizations
                - The mysterious script and attempts to decode it
                - How this ancient heritage connects to modern Haryana
                
                Present it as an engaging narrative that brings ancient history to life.
                """
            else:
                historical_sites = self.heritage_data[state]["historical_sites"]
                selected_site = random.choice(historical_sites)
                return base_context + f"""
                Create a historical narrative centered around {selected_site} in {state.title()}.
                The story should be 300-500 words and weave together:
                - Historical events and personalities
                - Cultural significance and traditions
                - Architectural marvels and their stories
                - The impact on modern {state} culture
                
                Make history come alive through storytelling.
                """
        
        else:  # random story
            elements = self.heritage_data[state]
            return base_context + f"""
            Create an original story inspired by the culture and heritage of {state.title()}.
            Incorporate elements such as:
            - Cultural traditions: {random.choice(elements['cultural_elements'])}
            - Historical context from the region
            - Local landscapes, festivals, or customs
            - Traditional values and wisdom
            
            The story should be 300-500 words, engaging, and culturally authentic.
            """

    def generate_story_content(self, state, story_type, language, custom_prompt=None):
        """Generate story content using Gemini"""
        try:
            prompt = self.generate_story_prompt(state, story_type, custom_prompt)
            
            if language != "english":
                prompt += f"\n\nIMPORTANT: Write the story in {language.title()} language."
            
            # Use a shorter response format for faster generation
            prompt += "\n\nPlease provide a concise story of 300-500 words."
            
            response = self.model.generate_content(prompt)
            story_text = response.text
            
            # Generate image prompts - pass story_text for better prompts
            image_prompt = f"""
            Based on this {state} cultural story, create 5 detailed visual scene descriptions for illustration:
            
            Story: {story_text[:800]}  # Limit length to avoid token issues
            
            Provide 5 distinct scenes as image prompts, each describing:
            - Setting and environment typical of {state}
            - Characters in traditional attire
            - Cultural elements and artifacts
            - Mood and atmosphere
            
            Format as: SCENE1: [description], SCENE2: [description], SCENE3: [description], SCENE4: [description], SCENE5: [description]
            """
            
            image_response = self.model.generate_content(image_prompt)
            image_prompts = self.parse_image_prompts(image_response.text, story_text, state)
            
            return story_text, image_prompts
            
        except Exception as e:
            logger.error(f"Error generating story: {str(e)}")
            return f"Error generating story: {str(e)}", []

    def parse_image_prompts(self, response_text, story_text, state):
        """Extract and enhance image prompts from the generated response"""
        prompts = []
        lines = response_text.split('\n')
        
        for line in lines:
            if any(line.strip().startswith(f"SCENE{i}:") for i in range(1, 6)):
                prompt = line.split(':', 1)[1].strip()
                prompts.append(prompt)
        
        # Fallback if parsing fails - create better prompts from story
        if len(prompts) < 5:
            # Extract key elements from story to create better prompts
            key_elements = self.extract_key_elements(story_text, state)
            visual_elements = self.heritage_data[state]["visual_elements"]
            
            # Create 5 prompts
            for i in range(5):
                prompts.append(
                    f"A scene from {state} showing {random.choice(visual_elements)}, {key_elements}, detailed, vibrant colors, cultural authenticity, scene {i+1}"
                )
        
        return prompts[:5]  # Return only 5 prompts

    def extract_key_elements(self, story_text, state):
        """Extract key cultural elements from story for better image prompts"""
        cultural_keywords = {
            "haryana": "wrestling, agriculture, folk dance, rural life, traditional attire, earthy tones",
            "punjab": "bhangra, golden temple, agriculture, vibrant festivals, Sikh culture, bright colors",
            "rajasthan": "forts, palaces, desert, camels, traditional jewelry, royal heritage, warm colors"
        }
        
        return cultural_keywords.get(state.lower(), "cultural heritage, traditional art, vibrant colors")

    def generate_images_free(self, prompts, state):
        """Generate images using multiple free methods with fallbacks - parallelized"""
        images = []
        futures = []
        
        # Submit all image generation tasks to thread pool
        for i, prompt in enumerate(prompts):
            future = self.executor.submit(self.generate_single_image, prompt, state, i+1)
            futures.append(future)
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                img = future.result()
                if img:
                    images.append(img)
                else:
                    # Create a placeholder if image generation failed
                    placeholder = self.create_placeholder_image(f"Scene: {state.title()} Heritage")
                    images.append(placeholder)
            except Exception as e:
                logger.error(f"Error in image generation: {str(e)}")
                placeholder = self.create_placeholder_image(f"Scene: {state.title()} Heritage")
                images.append(placeholder)
        
        return images
    
    def generate_single_image(self, prompt, state, scene_num):
        """Generate a single image using multiple fallback methods"""
        try:
            # Try Pollinations AI first (free and reliable)
            img = self.try_pollinations_ai(prompt, state)
            if img:
                return img
                
            # Try HuggingFace Inference API
            img = self.try_huggingface_inference(prompt, state, scene_num)
            if img:
                return img
                
            # Fallback to stylized placeholder
            logger.info(f"Using fallback image for prompt: {prompt}")
            img = self.create_stylized_placeholder(prompt, state, scene_num)
            return img
            
        except Exception as e:
            logger.error(f"Error creating image: {str(e)}")
            # Fallback to simple placeholder
            return self.create_placeholder_image(f"Scene {scene_num}: {state.title()} Heritage")

    def try_pollinations_ai(self, prompt, state):
        """Generate image using Pollinations AI (free service)"""
        try:
            # Enhance prompt with state-specific details
            enhanced_prompt = f"{prompt}, {state} culture, traditional, detailed, vibrant colors, high quality, 4k, photorealistic"
            
            # URL encode the prompt
            encoded_prompt = urllib.parse.quote(enhanced_prompt)
            
            # Pollinations AI endpoint with better parameters
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            
            # Add parameters for better quality
            url += "?nologo=true&quality=high&width=512&height=512&model=flux&seed=42"
            
            # Make request with timeout and error handling
            try:
                response = session.get(url, timeout=15)
                
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    return image
                else:
                    logger.warning(f"Pollinations AI error: {response.status_code}")
                    return None
                    
            except (requests.exceptions.RequestException, socket.error, ConnectionError) as e:
                logger.warning(f"Pollinations AI connection failed: {str(e)}")
                return None
                
        except Exception as e:
            logger.warning(f"Pollinations AI failed: {str(e)}")
            return None
    
    def try_huggingface_inference(self, prompt, state, scene_num):
        """Try to generate image using HuggingFace Inference API"""
        try:
            if HF_TOKEN == "your_huggingface_token_here":
                return None
                
            # Use a more reliable model
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            # Enhance prompt with state-specific details
            enhanced_prompt = f"{prompt}, {state} culture, traditional, detailed, vibrant colors, high quality"
            
            payload = {
                "inputs": enhanced_prompt,
                "parameters": {
                    "width": 512,
                    "height": 512,
                    "num_inference_steps": 20,  # Reduced for speed
                    "guidance_scale": 7.5
                }
            }
            
            try:
                response = session.post(API_URL, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    return image
                elif response.status_code == 503:
                    # Model is loading, skip and use fallback
                    logger.info("Model is loading, using fallback instead")
                    return None
                else:
                    logger.warning(f"HuggingFace API error: {response.status_code}")
                    return None
                    
            except (requests.exceptions.RequestException, socket.error, ConnectionError) as e:
                logger.warning(f"HuggingFace API connection failed: {str(e)}")
                return None
                
        except Exception as e:
            logger.warning(f"HuggingFace API failed: {str(e)}")
            return None

    def create_stylized_placeholder(self, prompt, state, scene_num):
        """Create a more elaborate placeholder image with state-specific styling"""
        state_colors = {
            "haryana": {"bg": (255, 223, 196), "fg": (139, 69, 19)},  # Earth tones
            "punjab": {"bg": (255, 248, 196), "fg": (255, 69, 0)},    # Vibrant yellow/orange
            "rajasthan": {"bg": (255, 196, 196), "fg": (178, 34, 34)} # Red tones
        }
        
        colors = state_colors.get(state.lower(), {"bg": (200, 200, 255), "fg": (0, 0, 139)})
        img = Image.new('RGB', (512, 512), color=colors["bg"])
        draw = ImageDraw.Draw(img)
        
        # Try to use a font
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
        
        # Draw state-specific pattern
        if state.lower() == "haryana":
            # Agricultural pattern
            for y in range(0, 512, 40):
                for x in range(0, 512, 40):
                    draw.rectangle([x, y, x+20, y+20], fill=colors["fg"], outline=None)
        elif state.lower() == "punjab":
            # Vibrant pattern
            for y in range(0, 512, 30):
                draw.line([(0, y), (512, y)], fill=colors["fg"], width=2)
        elif state.lower() == "rajasthan":
            # Royal pattern
            for x in range(0, 512, 50):
                draw.arc([x, 100, x+40, 140], 0, 180, fill=colors["fg"], width=3)
        
        # Add text
        title = f"{state.title()} Cultural Scene {scene_num}"
        bbox = draw.textbbox((0, 0), title, font=font_large)
        text_width = bbox[2] - bbox[0]
        draw.text(((512 - text_width) // 2, 50), title, font=font_large, fill=colors["fg"])
        
        # Add shortened prompt
        words = prompt.split()[:8]
        short_prompt = " ".join(words) + "..."
        bbox = draw.textbbox((0, 0), short_prompt, font=font_small)
        text_width = bbox[2] - bbox[0]
        draw.text(((512 - text_width) // 2, 100), short_prompt, font=font_small, fill=(0, 0, 0))
        
        # Add decorative element based on state
        if state.lower() == "haryana":
            draw.ellipse([206, 206, 306, 306], fill=(34, 139, 34), outline=(0, 100, 0))  # Green circle
        elif state.lower() == "punjab":
            draw.rectangle([206, 206, 306, 306], fill=(255, 215, 0), outline=(200, 150, 0))  # Golden square
        elif state.lower() == "rajasthan":
            draw.polygon([256, 206, 206, 306, 306, 306], fill=(178, 34, 34), outline=(139, 0, 0))  # Red triangle
        
        return img

    def create_placeholder_image(self, text):
        """Create a placeholder image with text"""
        img = Image.new('RGB', (512, 512), color=(70, 130, 180))
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (512 - text_width) // 2
        y = (512 - text_height) // 2
        
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        return img

    def generate_audio(self, text, language):
        """Generate audio using gTTS"""
        try:
            # Map language codes
            lang_code = self.languages.get(language, "en")
            
            # Limit text length to avoid TTS errors
            if len(text) > 3000:  # Reduced from 4000 for faster generation
                text = text[:3000] + "..."
            
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name
                
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            return None

    def create_video_with_captions(self, images, audio_path, story_text, language):
        """Create video combining images, audio, and captions"""
        if not MOVIEPY_AVAILABLE:
            return None
            
        try:
            if not audio_path or not os.path.exists(audio_path):
                return None
            
            # Load audio to get duration
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            
            # Calculate duration per image
            image_duration = duration / len(images)
            
            # Create video clips from images with captions
            video_clips = []
            temp_image_files = []  # Keep track of temp files for cleanup
            
            # Split story into segments for each image
            words = story_text.split()
            words_per_segment = max(1, len(words) // len(images))
            
            for i, img in enumerate(images):
                # Get text segment for this image
                start_word = i * words_per_segment
                end_word = start_word + words_per_segment
                if i == len(images) - 1:  # Last segment gets remaining words
                    segment_text = " ".join(words[start_word:])
                else:
                    segment_text = " ".join(words[start_word:end_word])
                
                # Limit caption length
                if len(segment_text) > 120:
                    segment_text = segment_text[:117] + "..."
                
                # Create image with caption using PIL
                img_with_caption = self.add_caption_to_image(img, segment_text)
                
                # Save image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                    img_with_caption.save(tmp_img.name, "JPEG", quality=95)
                    temp_image_files.append(tmp_img.name)
                    
                    # Create video clip from image
                    img_clip = ImageClip(tmp_img.name, duration=image_duration)
                    video_clips.append(img_clip)
            
            # Concatenate video clips
            video = concatenate_videoclips(video_clips, method="compose")
            
            # Set audio to the video
            final_video = video.set_audio(audio_clip)
            
            # Save video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                try:
                    final_video.write_videofile(
                        tmp_video.name, 
                        fps=24, 
                        verbose=False, 
                        logger=None,
                        codec='libx264',
                        audio_codec='aac',
                        threads=4,
                        preset='fast'
                    )
                    
                    # Cleanup temporary image files
                    for temp_file in temp_image_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                    
                    # Close clips to free memory
                    audio_clip.close()
                    for clip in video_clips:
                        clip.close()
                    final_video.close()
                    
                    return tmp_video.name
                    
                except Exception as write_error:
                    logger.error(f"Error writing video file: {write_error}")
                    return None
                
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            return None

    def add_caption_to_image(self, img, text):
        """Add caption to image using PIL (no ImageMagick dependency)"""
        # Convert to PIL Image if it's not already
        if not isinstance(img, Image.Image):
            pil_img = Image.fromarray(np.uint8(img))
        else:
            pil_img = img.copy()
        
        # Resize if needed
        if pil_img.size != (512, 512):
            pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Create a copy to draw on
        img_with_text = pil_img.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # Try to use a font
        try:
            # Try different font paths
            font_paths = [
                "arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 20)
                    break
                except:
                    continue
            
            # Fallback to default font if no font found
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (bottom of image)
        padding = 10
        max_width = 512 - (padding * 2)
        
        # Wrap text to fit within image width
        lines = []
        words = text.split()
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            # Estimate text width (approximate)
            text_width = len(test_line) * 10  # Rough approximation
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to 3 lines maximum
        if len(lines) > 3:
            lines = lines[:2]
            lines[-1] = lines[-1][:50] + "..." if len(lines[-1]) > 50 else lines[-1]
        
        # Draw semi-transparent background for text
        line_height = 25
        text_bg_height = len(lines) * line_height + padding * 2
        draw.rectangle(
            [(0, 512 - text_bg_height), (512, 512)], 
            fill=(0, 0, 0, 180)  # Semi-transparent black
        )
        
        # Draw text
        y_position = 512 - text_bg_height + padding
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x_position = (512 - text_width) // 2
            
            # Draw text with shadow effect
            draw.text((x_position+1, y_position+1), line, font=font, fill=(0, 0, 0))
            draw.text((x_position, y_position), line, font=font, fill=(255, 255, 255))
            
            y_position += line_height
        
        return img_with_text

    def create_simple_slideshow(self, images, audio_path, story_text):
        """Create a simple slideshow as alternative to video (without MoviePy)"""
        if not images or not audio_path:
            return None
            
        try:
            # Create image paths for the HTML
            image_paths = []
            for i, img in enumerate(images):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                    img.save(tmp_img.name, "JPEG")
                    image_paths.append(tmp_img.name)
            
            # Create a simple HTML slideshow
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Story Slideshow</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        margin: 0;
                        padding: 20px;
                    }}
                    .slideshow-container {{
                        max-width: 800px;
                        position: relative;
                        margin: auto;
                        background: white;
                        border-radius: 15px;
                        overflow: hidden;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                    }}
                    .slide {{
                        display: none;
                        text-align: center;
                        padding: 20px;
                    }}
                    .slide.active {{
                        display: block;
                    }}
                    .slide img {{
                        max-width: 100%;
                        height: 400px;
                        object-fit: cover;
                        border-radius: 10px;
                    }}
                    .slide h2 {{
                        color: #333;
                        margin: 20px 0;
                    }}
                    .slide p {{
                        color: #666;
                        line-height: 1.6;
                        text-align: justify;
                        margin: 0 20px;
                    }}
                    .controls {{
                        text-align: center;
                        padding: 20px;
                        background: #f8f9fa;
                    }}
                    button {{
                        background: #667eea;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        margin: 0 5px;
                        border-radius: 5px;
                        cursor: pointer;
                    }}
                    button:hover {{
                        background: #5a6fd8;
                    }}
                    .audio-container {{
                        text-align: center;
                        padding: 20px;
                        background: #f8f9fa;
                    }}
                </style>
            </head>
            <body>
                <div class="slideshow-container">
                    <div class="audio-container">
                        <h3>🎵 Listen to the Story</h3>
                        <audio controls style="width: 100%; max-width: 400px;">
                            <source src="{audio_path}" type="audio/mpeg">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
            """
            
            # Add slides
            for i, img_path in enumerate(image_paths):
                # Read image as base64
                with open(img_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                html_content += f"""
                    <div class="slide" id="slide{i}">
                        <img src="data:image/jpeg;base64,{img_data}" alt="Scene {i+1}">
                        <h2>Scene {i+1}</h2>
                    </div>
                """
            
            html_content += """
                    <div class="controls">
                        <button onclick="changeSlide(-1)">❮ Previous</button>
                        <button onclick="toggleAutoplay()" id="autoplayBtn">▶️ Auto Play</button>
                        <button onclick="changeSlide(1)">Next ❯</button>
                    </div>
                </div>
                
                <script>
                    let currentSlide = 0;
                    let autoplay = false;
                    let autoplayInterval;
                    
                    function showSlide(n) {
                        const slides = document.getElementsByClassName('slide');
                        if (n >= slides.length) currentSlide = 0;
                        if (n < 0) currentSlide = slides.length - 1;
                        
                        for (let i = 0; i < slides.length; i++) {
                            slides[i].classList.remove('active');
                        }
                        slides[currentSlide].classList.add('active');
                    }
                    
                    function changeSlide(direction) {
                        currentSlide += direction;
                        showSlide(currentSlide);
                    }
                    
                    function toggleAutoplay() {
                        const btn = document.getElementById('autoplayBtn');
                        if (autoplay) {
                            clearInterval(autoplayInterval);
                            btn.textContent = '▶️ Auto Play';
                            autoplay = false;
                        } else {
                            autoplayInterval = setInterval(() => {
                                currentSlide++;
                                showSlide(currentSlide);
                            }, 5000);
                            btn.textContent = '⏸️ Pause';
                            autoplay = true;
                        }
                    }
                    
                    // Initialize
                    showSlide(0);
                </script>
            </body>
            </html>
            """
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as tmp_file:
                tmp_file.write(html_content)
                
                # Cleanup image temp files
                for img_path in image_paths:
                    try:
                        os.unlink(img_path)
                    except:
                        pass
                        
                return tmp_file.name
                
        except Exception as e:
            logger.error(f"Error creating slideshow: {str(e)}")
            return None

    def generate_complete_story(self, state, story_type, language, custom_prompt=None):
        """Generate complete story with text, audio, images, and video"""
        try:
            # Generate story content
            story_text, image_prompts = self.generate_story_content(state, story_type, language, custom_prompt)
            
            if story_text.startswith("Error"):
                return story_text, None, [], None
            
            # Generate images using free method (parallelized)
            images = self.generate_images_free(image_prompts, state)
            
            # Generate audio
            audio_path = self.generate_audio(story_text, language)
            
            # Generate video (if MoviePy is available) or slideshow
            if MOVIEPY_AVAILABLE:
                video_path = self.create_video_with_captions(images, audio_path, story_text, language)
            else:
                video_path = self.create_simple_slideshow(images, audio_path, story_text)
            
            return story_text, audio_path, images, video_path
            
        except Exception as e:
            logger.error(f"Error generating complete story: {str(e)}")
            return f"Error generating complete story: {str(e)}", None, [], None

# Create the Storyteller instance
storyteller = CulturalStoryteller()

# Gradio Interface Functions
def generate_story_interface(state, story_type, language, custom_prompt):
    """Interface function for Gradio"""
    if not state:
        return "Please select a state.", None, None, None, None, None, None, None, None
    
    # Clean custom prompt
    custom_prompt = custom_prompt.strip() if custom_prompt else None
    if custom_prompt and len(custom_prompt) < 10:
        custom_prompt = None
    
    story_text, audio_path, images, video_path = storyteller.generate_complete_story(
        state.lower(), story_type, language.lower(), custom_prompt
    )
    
    # Ensure we have 5 images (pad with None if necessary)
    while len(images) < 5:
        images.append(None)
    
    # Return individual components
    return (story_text, audio_path, 
            images[0], images[1], images[2], images[3], 
            images[4], video_path)

def get_random_heritage_info(state):
    """Get random heritage information for the selected state"""
    if not state:
        return "Please select a state to learn about its heritage."
    
    state_key = state.lower()
    if state_key not in storyteller.heritage_data:
        return "Heritage information not available for this state."
    
    heritage = storyteller.heritage_data[state_key]
    
    info = f"## Heritage of {state.title()}\n\n"
    info += f"**Popular Folk Tales:**\n"
    for tale in heritage['folk_tales'][:3]:
        info += f"• {tale}\n"
    
    info += f"\n**Historical Sites:**\n"
    for site in heritage['historical_sites'][:3]:
        info += f"• {site}\n"
    
    info += f"\n**Cultural Elements:**\n"
    for element in heritage['cultural_elements'][:3]:
        info += f"• {element}\n"
    
    return info

# Create Gradio Interface
with gr.Blocks(title="AI Cultural Storyteller - Heritage of Northern India", theme=gr.themes.Soft()) as interface:
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #FF6B35, #F7931E, #FFD23F);">
        <h1 style="color: white; margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🏛️ AI Cultural Storyteller 🏛️
        </h1>
        <h2 style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">
            Preserving the Heritage of Haryana, Punjab & Rajasthan
        </h2>
    </div>
    """)
    
    gr.Markdown("""
    ## 📚 About This Project
    This AI storyteller specializes in the rich cultural heritage of Northern India, focusing on:
    - **Haryana**: Including Rakhigarhi (largest Indus Valley site in India)
    - **Punjab**: Land of five rivers and Sikh heritage
    - **Rajasthan**: Desert kingdom with royal traditions
    
    Generate authentic folk tales, historical narratives, and cultural stories with multilingual support!
    """)
    
    with gr.Tab("🎭 Story Generator"):
        with gr.Row():
            with gr.Column(scale=1):
                state_dropdown = gr.Dropdown(
                    choices=["Haryana", "Punjab", "Rajasthan"],
                    label="Select State",
                    value="Haryana"
                )
                
                story_type_dropdown = gr.Dropdown(
                    choices=["folk_tale", "historical", "random"],
                    label="Story Type",
                    value="folk_tale"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=["English", "Hindi", "Punjabi", "Gujarati", "Marathi", "Bengali"],
                    label="Language",
                    value="English"
                )
                
                custom_prompt_input = gr.Textbox(
                    label="Custom Story Prompt (Optional)",
                    placeholder="E.g., Tell me a story about a brave farmer during harvest festival...",
                    lines=3
                )
                
                generate_btn = gr.Button("🎪 Generate Story", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                heritage_info = gr.Markdown(label="Heritage Information")
                
                # Update heritage info when state changes
                state_dropdown.change(
                    fn=get_random_heritage_info,
                    inputs=[state_dropdown],
                    outputs=[heritage_info]
                )
        
        with gr.Tab("📖 Story Output"):
            story_text_output = gr.Textbox(
                label="Generated Story",
                lines=20,
                max_lines=30
            )
        
        with gr.Tab("🎵 Audio"):
            audio_output = gr.Audio(
                label="Story Audio",
                type="filepath"
            )
        
        with gr.Tab("🖼️ Images"):
            with gr.Row():
                image_output_1 = gr.Image(label="Scene 1")
                image_output_2 = gr.Image(label="Scene 2")
                image_output_3 = gr.Image(label="Scene 3")
            with gr.Row():
                image_output_4 = gr.Image(label="Scene 4")
                image_output_5 = gr.Image(label="Scene 5")
        
        with gr.Tab("🎬 Video/Slideshow"):
            if MOVIEPY_AVAILABLE:
                video_output = gr.Video(label="Story Video with Captions")
            else:
                video_output = gr.HTML(label="Interactive Story Slideshow")
                gr.Markdown("💡 **Note**: MoviePy not installed. Showing interactive slideshow instead of video.")
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_story_interface,
            inputs=[state_dropdown, story_type_dropdown, language_dropdown, custom_prompt_input],
            outputs=[story_text_output, audio_output, 
                    image_output_1, image_output_2, image_output_3, 
                    image_output_4, image_output_5, video_output]
        )
    
    with gr.Tab("📋 Heritage Database"):
        gr.Markdown("""
        ## 🏛️ Cultural Heritage Database
        
        ### Haryana
        - **Rakhigarhi**: Largest Indus Valley Civilization site in India
        - **Kurukshetra**: Sacred land of the Bhagavad Gita
        - Rich tradition of wrestling, folk dances, and agricultural festivals
        
        ### Punjab
        - **Golden Temple**: Spiritual center of Sikhism
        - Famous love legends: Heer-Ranjha, Sassi-Punnu, Sohni-Mahiwal
        - Vibrant culture of Bhangra, Giddha, and literary traditions
        
        ### Rajasthan
        - **Desert Kingdom**: Land of forts, palaces, and royal traditions
        - Legendary figures: Maharana Pratap, Meera Bai, Padmavati
        - Rich folk music, dance, and colorful festivals
        """)
    
    with gr.Tab("⚙️ Setup Instructions"):
        gr.Markdown(f"""
        ## 🔧 Setup Instructions
        
        To run this storyteller, you need to configure the following API keys:
        
        1. **Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/)
        2. **HuggingFace Token** (optional): For better image generation
        
        ### Installation Requirements:
        ```bash
        # Basic requirements
        pip install gradio google-generativeai gtts pillow requests
        
        # For video generation (recommended)
        pip install moviepy opencv-python
        ```
        
        **Note**: This version uses free image generation services that work without GPU.
        
        ### Environment Variables:
        Set these in your environment or create a `.env` file:
        ```
        GEMINI_API_KEY=your_gemini_api_key_here
        HF_TOKEN=your_huggingface_token_here  # Optional
        ```
        
        ### Features:
        - ✅ Multi-language support (Hindi, English, Punjabi, etc.)
        - ✅ Text-to-speech audio generation
        - ✅ AI-generated images using Pollinations AI (free)
        - ✅ Video creation with captions
        - ✅ Cultural authenticity and historical accuracy
        - ✅ Custom story prompts
        - ✅ Heritage database integration
        
        **Note**: This version uses Pollinations AI for free image generation which works reliably without GPU.
        """)
    
    # Initialize with heritage info
    interface.load(
        fn=get_random_heritage_info,
        inputs=[gr.State("Haryana")],
        outputs=[heritage_info]
    )

# Launch the interface
if __name__ == "__main__":
    # Add some startup heritage info
    print("🏛️ AI Cultural Storyteller - Heritage of Northern India")
    print("📚 Specializing in Haryana, Punjab, and Rajasthan")
    print("🎭 Folk tales, Historical narratives, and Cultural stories")
    print("🌍 Multilingual support for regional reach")
    print("🖼️ Using Pollinations AI for free image generation")
    
    # Check for API keys
    if GEMINI_API_KEY == "your_gemini_api_key_here":
        print("\n⚠️  WARNING: Please set your GEMINI_API_KEY environment variable!")
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to False to avoid connection issues
        debug=False   # Set to False to reduce verbosity
    )