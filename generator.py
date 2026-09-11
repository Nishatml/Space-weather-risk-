def generate_incident_report(client, metrics):
    """Generates structured 4-section technical incident advisory via Groq GPT-OSS 20B"""
    if not client:
        return "Groq API Key unavailable. Cannot generate LLM report."
    
    prompt = f"""
    You are an expert Space Weather Systems Engineer. Generate a structured 4-section technical incident report based on the following DSCOVR spacecraft telemetry and ML forecasts:
    
    - Predicted Kp Index: {metrics['kp_index']}
    - Storm Probability: {metrics['storm_prob']}%
    - IMF Bz Component: {metrics['bz']} nT
    - Solar Wind Speed: {metrics['speed']} km/s
    - Proton Density: {metrics['density']} p/cm³
    
    Sections required:
    1. Executive Summary & Threat Level Assessment
    2. Power Grid & Geomagnetically Induced Current (GIC) Impact
    3. Satellite Drag & High-Frequency (HF) Radio Communications
    4. Recommended Operational Mitigation Steps
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content