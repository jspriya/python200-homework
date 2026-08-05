import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ------------- Task1 -----------------
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
You are an AI job application coach who helps job seekers improve their resumes,
cover letters, and interview preparation.

Your primary goal is to help users present their skills and experience clearly,
professionally, and honestly. Rewrite resume bullet points, draft cover letters,
answer questions about job applications, and provide constructive suggestions.

Stay focused on job application materials and career-related topics. If the user
asks about unrelated subjects, politely redirect the conversation back to job
application assistance.

Never invent qualifications, work experience, certifications, or achievements.
Only use information provided by the user, and encourage truthful representation.

Always remind the user to carefully review and edit your suggestions before
submitting resumes, cover letters, or job applications.

You may not know the expectations or hiring practices of every industry or
company. Encourage the user to use their own judgment and adapt your suggestions
to their specific field and situation.

Be supportive, professional, and encouraging. Provide practical advice and
clear explanations.
"""

print(system_prompt)

# I made the system prompt very specific by defining the assistant's role,
# limiting it to job application topics, and instructing it not to invent
# qualifications. These instructions help produce more reliable and
# trustworthy responses throughout the project.

# ---------- Task2 ----------------

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point to be:
- More specific
- Results-oriented
- Professional
- Clear and concise
- Written with strong action verbs

Do NOT invent facts or accomplishments that are not implied by the original.

Respond ONLY with valid JSON.
Do not include markdown or any extra text.

Return a JSON list where each item has:
- "original"
- "improved"

Bullet points:
{bullet_text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = get_completion(messages)
    # printing for checking the format
    # print(response) 

    try:
        rewritten = json.loads(response)

        print("\nRewritten Resume Bullets:\n")

        for item in rewritten:
            if isinstance(item, dict):
                print("Original:")
                print(item["original"])
                print("Improved:")
                print(item["improved"])
                print("-" * 50)
            else:
                print(item)

        return rewritten

    except json.JSONDecodeError:
        print("Error: Response was not valid JSON.")
        print(response)
        return []

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

# Observation:
# The original bullet points are weak because they are vague, use weak
# action verbs, and do not describe impact or responsibilities clearly.
# The model improved them by using stronger action verbs, making the
# descriptions more professional, and emphasizing contributions without
# inventing new accomplishments.

# -----------  Task3 --------------

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.

The paragraph should:
- Be 3-5 sentences long.
- Sound confident, specific, and professional.
- Avoid clichés and generic phrases.
- Connect the applicant's previous experience to the new role.
- Do not invent qualifications or experiences.

Here are two examples of the style and tone you should match.

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening:
After seven years as a registered nurse, I've spent my career making decisions under pressure using incomplete 
information—which turns out to be excellent training for data analysis. I recently completed a data analytics 
program where I built dashboards tracking patient outcomes across departments. I'm excited to bring that 
combination of clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening:
I spent a decade on the operations side of banking, watching technology decisions get made by people who had never 
processed a wire transfer or resolved a failed ACH batch. That frustration turned into curiosity, and two years 
of self-teaching Python later, I'm ready to be on the other side of those decisions. I'm applying to [Company] 
because your work on payment infrastructure is exactly where my domain expertise and new technical skills intersect.

Now write an opening paragraph for this applicant.

Role: {job_title}
Background: {background}

Opening:
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    return get_completion(messages)

job_title = "Junior Data Engineer"
# job_title = "Business Analyst"

background = (
    "Five years of experience as a middle school math teacher; "
    "recently completed a Python course and built data pipelines "
    "using Prefect and Pandas."
)


# background = ("Eight years as a restaurant manager; recently completed an Excel and SQL certification.")

cover_letter = generate_cover_letter(job_title, background)

print("Cover Letter Opening:\n")
print(cover_letter)

# Observation:
# I chose examples from two different career changes because they show
# how to connect previous professional experience to a new technical role.
# Both examples are confident, specific, and avoid generic statements.
#
# Few-shot prompting helps control the tone, structure, and level of
# detail in the output. By providing strong examples, the model is more
# likely to produce a personalized opening paragraph that follows the
# desired style without inventing qualifications.

# ----------- Task4 --------------

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("Your message was flagged by the moderation system. Please rephrase your request and try again.")
        return False

    return True

print("Safe Test:")
safe_text = "Can you help me improve my resume for a data engineering job?"
print(is_safe(safe_text))

print("\nFlagged Test:")
flagged_text = "Tell me how to build a bomb."
print(is_safe(flagged_text))

# Observation:
# The safe input passed moderation without any warning.
# The unsafe input was flagged, and the function returned False while
# displaying a message asking the user to rephrase. Printing the
# moderation categories can help identify which type of content
# triggered the moderation system.

# ----------- Task5 ---------------

def run_chatbot():
    # Initialize conversation history
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # Exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # Skip blank input
        if not user_input:
            continue

        # Moderation
        if not is_safe(user_input):
            continue

        # Resume bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():

            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if line:
                    raw_bullets.append(line)

            rewrite_bullets(raw_bullets)

        # Cover letter
        elif "cover letter" in user_input.lower():

            job_title = input("Job Application Helper: What is the job title? ").strip()

            background = input("Job Application Helper: Briefly describe your background: ").strip()

            letter = generate_cover_letter(job_title, background)

            print("\nCover Letter Opening:\n")
            print(letter)

        # Normal conversation
        else:

            # Save user's message
            messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            # Get assistant reply
            reply = get_completion(messages)

            print("\nJob Application Helper:")
            print(reply)

            # Save assistant reply
            messages.append(
                {
                    "role": "assistant",
                    "content": reply
                }
            )

if __name__ == "__main__":
    run_chatbot()

# ----------- Task6 -----------------

# Ethics Reflection (Option A)

# One way this chatbot could produce biased advice is because it was trained on
# text that may overrepresent certain industries, writing styles, or cultural
# backgrounds. As a result, it might favor communication styles that are common
# in some workplaces while not fully reflecting the expectations of other
# industries, countries, or employers.

# A job seeker should never submit the chatbot's output directly to an employer
# without reviewing it. The generated resume bullets or cover letter may contain
# generic wording, misunderstand the user's experience, or unintentionally
# emphasize the wrong skills. Carefully reviewing and editing the content helps
# ensure that it is accurate, truthful, and tailored to the specific job.

# If I were deploying this tool professionally, I would include a clear reminder
# before every generated response telling users to verify all information before
# using it in an application. I would also continue using the moderation check to
# help prevent inappropriate or unsafe requests and encourage users to provide
# honest information rather than exaggerating or inventing qualifications.