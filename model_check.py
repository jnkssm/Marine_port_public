# test_ollama.py
import requests

prompt = """

You are an AI ship agent optimising your port-arrival schedule to maximise rewards.

ENVIRONMENT PARAMETERS (set by the simulation operator):
• Total ships in simulation : 10
• Port capacity threshold   : 60%  (6 ships)
• A day is CONGESTED when   : arrivals > 6 ships
• A day is NON-CONGESTED    : arrivals ≤ 6 ships

REWARD SYSTEM:
• Arrive on a NON-CONGESTED day   : +1  (BEST)
• Arrive on a CONGESTED day       : -1  (WORST)
• Stay home on a CONGESTED day    : +0.5 (GOOD)
• Stay home on a NON-CONGESTED day:  0  (NEUTRAL)

MY CURRENT STRATEGY PERFORMANCE (sorted best→worst, includes recent rewards):
  Rank 1: [0, 0, 1, 0, 1, 0, 1]  →  [Wed, Fri, Sun]  Q=9.064
  Rank 2: [0, 1, 1, 0, 0, 1, 1]  →  [Tue, Wed, Sat, Sun]  Q=5.270
  Rank 3: [1, 0, 0, 1, 0, 0, 1]  →  [Mon, Thu, Sun]  Q=4.414
  Rank 4: [1, 1, 0, 0, 0, 1, 1]  →  [Mon, Tue, Sat, Sun]  Q=2.000
  Rank 5: [1, 0, 0, 1, 1, 0, 0]  →  [Mon, Thu, Fri]  Q=2.000
  Rank 6: [1, 0, 1, 0, 0, 1, 1]  →  [Mon, Wed, Sat, Sun]  Q=0.934
  Rank 7: [0, 1, 0, 0, 1, 1, 0]  →  [Tue, Fri, Sat]  Q=0.887
  Rank 8: [1, 1, 0, 0, 0, 1, 0]  →  [Mon, Tue, Sat]  Q=0.525
  Rank 9: [1, 1, 0, 0, 1, 0, 0]  →  [Mon, Tue, Fri]  Q=0.525
  Rank 10: [1, 0, 0, 1, 0, 1, 1]  →  [Mon, Thu, Sat, Sun]  Q=0.500
  Rank 11: [1, 0, 1, 0, 1, 0, 0]  →  [Mon, Wed, Fri]  Q=-0.020
  Rank 12: [0, 1, 0, 1, 0, 1, 0]  →  [Tue, Thu, Sat]  Q=-0.106
  Last-week rewards : Mon: +0.0 | Tue: +0.0 | Wed: +3.0 | Thu: +0.0 | Fri: +3.0 | Sat: +0.0 | Sun: +3.0
  Last-week avg     : 1.286

  EXISTING strategies (do NOT repeat these):
    [1, 0, 1, 0, 1, 0, 0]
    [0, 1, 0, 1, 0, 1, 0]
    [1, 0, 0, 1, 0, 0, 1]
    [0, 0, 1, 0, 1, 0, 1]
    [1, 1, 0, 0, 0, 1, 0]
    [1, 0, 0, 1, 0, 1, 1]
    [1, 1, 0, 0, 1, 0, 0]
    [0, 1, 1, 0, 0, 1, 1]
    [1, 0, 1, 0, 0, 1, 1]
    [0, 1, 0, 0, 1, 1, 0]
    [1, 1, 0, 0, 0, 1, 1]
    [1, 0, 0, 1, 1, 0, 0]

Historical total arrivals per day (from simulation start): [10, 0, 10, 0, 10, 0, 0, 0, 10, 0, 10, 0, 10, 0, 10, 0, 0, 10, 0, 0, 10, 0, 0, 10, 0, 10, 0, 10, 10, 9, 1, 0, 0, 10, 1, 5, 4, 6, 4, 6, 5, 5, 5, 4, 6, 4, 6, 5, 5, 5, 4, 6, 4, 6, 5, 5, 5, 4, 6, 4, 6, 5, 5, 5, 4, 6, 4, 6, 5, 5, 5, 4, 6, 4, 6, 5, 5, 7, 4, 5, 6, 4, 4, 4, 5, 5, 6, 4, 5, 6, 5, 4, 5, 5, 4, 6, 6, 4, 5, 4, 6, 4, 6, 5, 5, 5, 4, 6, 4, 6, 5, 5, 5, 4, 6, 4, 6, 5, 5, 5, 4, 6, 4, 6, 5, 5]

YOUR TASK:
Suggest ONE new weekly schedule that is DIFFERENT from all existing strategies
listed above and could improve my rewards by spreading attendance on days that
are statistically less congested.

REQUIREMENTS:
1. Attend exactly 3 or 4 days per week.
2. The new strategy must NOT match any strategy in the EXISTING list above.
3. Prefer days that historically had lower attendance in the arrival history.
4. Avoid patterns from low-ranked strategies.

STRICT OUTPUT FORMAT — respond with ONLY a Python list of 7 binary integers (1=attend, 0=skip).
Example: [1,0,1,0,1,0,0]
Do NOT include any explanation, text, or punctuation outside the list.

"""

try:
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama2',
            'prompt': prompt,
            'stream': False,
            'temperature': 0.1,  # Lower temperature for more consistent output
            'max_tokens': 50
        },
        timeout=200  # Longer timeout for complex prompts
    )
    
    print(response)
    
    if response.status_code == 200:
        result = response.json()
        answer = result['response'].strip()
        print("Raw response:")
        print(answer)
        print("\n" + "="*50)
        
        # Try to extract just the list if there's extra text
        import re
        list_pattern = r'\[[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],[0-1]\]'
        match = re.search(list_pattern, answer)
        if match:
            print("\nExtracted strategy:")
            print(match.group())
        else:
            print("\nCould not extract strategy list from response")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("Error: Cannot connect to Ollama. Make sure Ollama is running:")
    print("  ollama serve")
except Exception as e:
    print(f"Error: {e}")