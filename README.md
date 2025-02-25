A good attempt on fine tuning LLAMA-3-1b using QLoRA and a synthetic dataset. 

Model available on Hugging Face: https://huggingface.co/Jiexing1028/llama-3-1b-workout
Dataset: https://huggingface.co/datasets/Jiexing1028/workout-plan

# inference with the fine tuned model

messages = [
    {"role": "system", "content": "you are a helpful assistant in generating personalised workout plans."},
    {"role": "user", "content": "I'm 25 years old and have severe elbow injury, what exercises should I avoid to prevent further injury?"},
]


## Recommended Workouts

Based on your injury history (severe elbow), your age (21–24 years) and body frame (low muscle mass), these workouts have been designed to help you improve muscle strength without aggravating your existing injury.

1. **Arm Curl**: A basic exercise for strengthening weakened elbows. Start by holding a weight (e.g., dumbbell or resistance band). Slowly bring the arm towards your shoulder, with each rep, aiming to fully relax the forearm muscles at the end of the movement.
2. **Bench Press**: This is a great exercise to improve core strength and abdominal muscles. Sit on the bench, lower yourself down until your elbows form 90-degree angles. Slowly raise yourself back up, keeping your arms parallel to the ground.
3. **Push-Up**: One of the most effective exercises to build muscle strength while avoiding damage to damaged tissue. While lying flat on the floor or an elevated surface, start with a push-up stance (your arms will be perpendicular to your body). Keeping your core braced, slowly perform one push-up then come back to standing. Aim for three sets of eight reps (
