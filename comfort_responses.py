comfort_map = {
    "anxiety": {
        "comfort": "Breathe… you're okay. You're overwhelmed, not weak 💛",
        "guide": "• Slow breaths\n• Step away from the trigger\n• Ground yourself with 5-4-3-2-1",
        "routine": "Take a 3-minute pause. Inhale for 4, hold for 2, exhale for 6.",
        "song": "‘Weightless’ – Marconi Union",
        "hobby": "Try doodling shapes. Your mind calms when your hands move.",
    },
    "depression": {
        "comfort": "You’re tired, not broken. I’m right here with you 💛",
        "guide": "• Get sunlight\n• Take a warm shower\n• Text one safe person",
        "routine": "Sit up, roll your shoulders back, sip water — tiny resets count.",
        "song": "‘Liability’ — Lorde",
        "hobby": "Journal one sentence about how you feel.",
    },
    "suicidal": {
        "comfort": "You’re hurting deeply, but you’re not alone. Stay with me right now 💛",
        "guide": "• Don’t isolate\n• Call a trusted person\n• Avoid sharp objects or unsafe places",
        "routine": "Place your hand on your chest. Feel that? You're still here. Stay.",
        "song": "‘Fix You’ — Coldplay",
        "hobby": "Hold something soft. Ground your senses.",
    },
    "neutral": {
        "comfort": "I hear you. Tell me more.",
        "guide": "You're doing okay. I'm listening.",
        "routine": "",
        "song": "",
        "hobby": "",
    }
}

def get_response(label):
    return comfort_map.get(label, comfort_map["neutral"])
