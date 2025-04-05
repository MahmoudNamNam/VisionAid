import base64
import pygame
import io
import os

def save_base64_audio_to_mp3(base64_string, output_filename="output.mp3"):
    """
    Decodes a base64 audio string and saves it as an MP3 file.

    Args:
        base64_string (str): The base64 encoded audio data.
        output_filename (str, optional): The name of the MP3 file to save.
                                         Defaults to "output.mp3".
    """
    try:
        audio_bytes = base64.b64decode(base64_string)
        with open(output_filename, "wb") as f:
            f.write(audio_bytes)
        print(f"Audio saved as {output_filename}")
    except Exception as e:
        print(f"Error saving audio: {e}")

audio_base64_string =''

audio_bytes = base64.b64decode(audio_base64_string)

# Initialize Pygame Mixer
pygame.mixer.init()

try:
    # Load the audio data into a Pygame Sound object
    sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))

    # Play the sound
    sound.play()
    print("Playing audio...")

    # Keep the script running while the sound is playing
    while pygame.mixer.get_busy():
        pygame.time.delay(100)

except pygame.error as e:
    print(f"Pygame error during playback: {e}")
    print("Pygame might not support this audio format directly.")
    print("Try saving as MP3 and playing with a dedicated MP3 player.")

finally:
    # Quit Pygame Mixer
    pygame.mixer.quit()

# Save the audio as an MP3 file
save_base64_audio_to_mp3(audio_base64_string, "arabic_speech.mp3")