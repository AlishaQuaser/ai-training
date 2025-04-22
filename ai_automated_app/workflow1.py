import os
import json
import re
import hashlib
import random
import time
import numpy as np
from dotenv import load_dotenv
from llama_index.llms.mistralai import MistralAI
from mistralai import Mistral
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.mistralai import MistralAIEmbedding
import nest_asyncio
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()
nest_asyncio.apply()

PROFILE_INDEX_DIR = "./profile_index"
CACHE_DIR = "./query_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5)
)
def api_call_with_retry(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if "rate limit" in str(e).lower():
            time.sleep(random.uniform(2, 5))
            raise
        else:
            raise


def get_cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()


def get_cached_response(text):
    cache_key = get_cache_key(text)
    cache_file = f"{CACHE_DIR}/{cache_key}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def cache_response(text, response):
    cache_key = get_cache_key(text)
    cache_file = f"{CACHE_DIR}/{cache_key}.json"
    with open(cache_file, "w") as f:
        json.dump(response, f)


def setup_mistral_llm():
    return MistralAI(model="mistral-large-latest")


def setup_mistral_client():
    return Mistral()

def extract_json(text):
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Could not parse JSON", "raw_text": text}


def get_profiles_from_database():
    """
    Get the profiles data from the database with only the needed fields
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("You need to set MONGO_URI in your environment variables.")
        return []

    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri)
        db = client["app-dev"]
        collection = db["profiles"]

        fields_to_retrieve = {
            "firstName": 1,
            "lastName": 1,
            "areaOfExpertise": 1,
            "currentLocation": 1,
            "carrierSummary": 1,
            "education": 1,
            "experience": 1,
            "highlightedSkills": 1
        }

        profiles = list(collection.find({}, fields_to_retrieve))
        if not profiles:
            print("No profiles found in the database.")
            return []

        for profile in profiles:
            profile["_id"] = str(profile["_id"])

        print(f"Fetched {len(profiles)} profiles from the database")
        return profiles
    except Exception as e:
        print(f"Error fetching profiles from database: {e}")
        return []


def encode_text(text, mistral_client):
    """
    Generate embeddings for the input text using Mistral API.
    """
    response = mistral_client.embeddings(
        model="mistral-embed",
        input=text
    )
    return np.array(response.data[0].embedding)


class ParseJobDescriptionEvent(Event):
    job_description: str


class ExtractRequirementsEvent(Event):
    requirements: dict


class RankCandidatesEvent(Event):
    candidates: list


class JobMatchingWorkflow(Workflow):
    profiles = None
    llm = None
    mistral_client = None
    index = None

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> ParseJobDescriptionEvent:
        if not ev.job_description:
            raise ValueError("No job description provided")

        self.llm = setup_mistral_llm()
        self.mistral_client = setup_mistral_client()

        mistral_embed = MistralAIEmbedding(model_name="mistral-embed")
        mistral_llm = MistralAI(model="mistral-large-latest")

        Settings.embed_model = mistral_embed
        Settings.llm = mistral_llm

        try:
            storage_context = StorageContext.from_defaults(persist_dir=PROFILE_INDEX_DIR)
            self.index = load_index_from_storage(storage_context=storage_context)
            print(f"Successfully loaded index from {PROFILE_INDEX_DIR}")
        except Exception as e:
            print(f"Error loading index: {e}")
            raise

        self.profiles = get_profiles_from_database()
        print(f"Working with {len(self.profiles)} profiles")

        return ParseJobDescriptionEvent(job_description=ev.job_description)

    @step
    async def parse_job_description(self, ctx: Context, ev: ParseJobDescriptionEvent) -> ExtractRequirementsEvent:
        print(f"Parsing job description: {ev.job_description[:100]}...")
        cache_key = f"jd_{get_cache_key(ev.job_description)}"
        cached_requirements = get_cached_response(cache_key)

        if cached_requirements:
            print("Using cached job requirements analysis")
            requirements = cached_requirements
        else:
            print("Analyzing job description with LLM")
            prompt = f"""
            Analyze this job description and extract the key requirements into distinct categories:
            - Technical skills required
            - Experience level needed
            - Education requirements
            - Soft skills required
            - Any other important qualifications

            Return as a JSON object with these categories as keys.

            Job Description:
            {ev.job_description}

            Return JSON only, no additional text.
            """

            await asyncio.sleep(1)
            raw_json = api_call_with_retry(self.llm.complete, prompt)
            requirements = extract_json(raw_json.text)
            print(f"Extracted requirements: {json.dumps(requirements, indent=2)}")
            cache_response(cache_key, requirements)

        await ctx.set("job_requirements", requirements)
        await ctx.set("job_description", ev.job_description)
        await ctx.set("candidate_results", [])

        return ExtractRequirementsEvent(requirements=requirements)

    @step
    async def query_candidates_per_requirement(self, ctx: Context, ev: ExtractRequirementsEvent) -> RankCandidatesEvent:
        job_description = await ctx.get("job_description")

        query_engine = self.index.as_query_engine(llm=self.llm)
        response = query_engine.query(job_description)

        retrieved_nodes = response.source_nodes
        if not retrieved_nodes:
            print("No matching profiles found")
            return RankCandidatesEvent(candidates=[])

        print(f"Found {len(retrieved_nodes)} matching profile nodes")

        all_matched_candidates = {}
        for node in retrieved_nodes:
            profile_text = node.node.text

            name_match = re.search(r"Name: (.*?)[\n]", profile_text)
            if not name_match:
                continue

            full_name = name_match.group(1).strip()
            first_name, *last_name_parts = full_name.split()

            matching_profile = None
            for profile in self.profiles:
                db_first_name = profile.get('firstName', '')
                db_last_name = profile.get('lastName', '')
                if first_name.lower() == db_first_name.lower() and any(part.lower() == db_last_name.lower() for part in last_name_parts):
                    matching_profile = profile
                    break

            if not matching_profile:
                continue

            profile_id = matching_profile["_id"]
            similarity_score = node.score if hasattr(node, 'score') else 0.5

            if profile_id not in all_matched_candidates:
                all_matched_candidates[profile_id] = {
                    "profile": matching_profile,
                    "scores": {},
                    "avg_score": 0
                }

            requirements = ev.requirements
            for category, req_details in requirements.items():
                all_matched_candidates[profile_id]["scores"][category] = similarity_score

        candidates_list = []
        for profile_id, candidate_data in all_matched_candidates.items():
            scores = list(candidate_data["scores"].values())
            if scores:
                avg_score = sum(scores) / len(scores)
                candidate_data["avg_score"] = avg_score
                candidates_list.append(candidate_data)

        candidates_list.sort(key=lambda x: x["avg_score"], reverse=True)
        top_candidates = candidates_list[:10]

        print(f"Found {len(all_matched_candidates)} total unique candidates")
        print(f"Selected {len(top_candidates)} top candidates for ranking")

        await ctx.set("top_candidates", top_candidates)

        return RankCandidatesEvent(candidates=top_candidates)

    @step
    async def rank_candidates(self, ctx: Context, ev: RankCandidatesEvent) -> StopEvent:
        print(f"Ranking {len(ev.candidates)} candidates")

        if not ev.candidates:
            print("No candidates to rank, returning empty result")
            return StopEvent(result=[])

        job_requirements = await ctx.get("job_requirements")
        job_req_text = json.dumps(job_requirements, indent=2)

        candidate_texts = []
        for i, candidate in enumerate(ev.candidates):
            profile = candidate["profile"]
            scores = candidate["scores"]

            candidate_text = f"Candidate {i+1}:\n"
            candidate_text += f"Name: {profile.get('firstName', '')} {profile.get('lastName', '')}\n"
            candidate_text += f"Area of Expertise: {profile.get('areaOfExpertise', 'Not specified')}\n"

            location = profile.get('currentLocation', {})
            location_str = f"{location.get('city', '')}, {location.get('state', '')}, {location.get('country', '')}"
            candidate_text += f"Location: {location_str}\n"

            candidate_text += f"Career Summary: {profile.get('carrierSummary', 'Not provided')}\n"

            education_entries = profile.get('education', [])
            if education_entries:
                candidate_text += "Education:\n"
                for edu in education_entries:
                    degree = edu.get('degree', 'Not specified')
                    institute = edu.get('institute', 'Not specified')
                    start_date = edu.get('startDate', 'Unknown date')
                    candidate_text += f"- {degree} at {institute} ({start_date})\n"

            experience_entries = profile.get('experience', [])
            if experience_entries:
                candidate_text += "Experience:\n"
                for exp in experience_entries:
                    position = exp.get('position', 'Not specified')
                    company = exp.get('company', 'Not specified')
                    start_date = exp.get('startDate', 'Unknown date')
                    candidate_text += f"- {position} at {company} ({start_date})\n"

            highlighted_skills = profile.get('highlightedSkills', [])
            if highlighted_skills:
                candidate_text += "Highlighted Skills: "
                candidate_text += ", ".join([s.get('name', 'Unknown') for s in highlighted_skills]) + "\n"

            candidate_text += f"Scores by requirement: {json.dumps(scores)}\n"
            candidate_text += f"Average score: {candidate['avg_score']:.2f}\n"

            candidate_texts.append(candidate_text)

        candidates_str = "\n\n".join(candidate_texts)

        print("Sending ranking prompt to LLM")
        prompt = f"""
        You are a recruiting AI assistant. You need to evaluate candidates against job requirements.
        
        Job Requirements:
        {job_req_text}
        
        Candidates:
        {candidates_str}
        
        Rank the top 5 candidates in order of best match to the requirements. For each candidate provide:
        1. Rank position (1-5)
        2. A match score from 0-100
        3. A brief explanation of why they ranked in this position
        
        Return as a JSON array where each item has the format:
        {{
            "rank": [1-5],
            "candidate_number": [candidate number from the list],
            "match_score": [0-100],
            "explanation": "[brief explanation]"
        }}
        
        Return JSON only, no additional text.
        """

        await asyncio.sleep(2)
        raw_json = api_call_with_retry(self.llm.complete, prompt)
        print(f"Received ranking response from LLM: {raw_json.text[:200]}...")

        ranking_results = extract_json(raw_json.text)
        print(f"Parsed ranking results: {json.dumps(ranking_results, indent=2)}")

        if not isinstance(ranking_results, list):
            for key in ranking_results:
                if isinstance(ranking_results[key], list):
                    ranking_results = ranking_results[key]
                    break

        final_results = []
        for rank_info in ranking_results:
            if "candidate_number" in rank_info:
                candidate_idx = rank_info["candidate_number"] - 1
                if 0 <= candidate_idx < len(ev.candidates):
                    candidate = ev.candidates[candidate_idx]
                    final_results.append({
                        "rank": rank_info["rank"],
                        "profile": candidate["profile"],
                        "match_score": rank_info["match_score"],
                        "explanation": rank_info["explanation"]
                    })

        final_results.sort(key=lambda x: x["rank"])
        print(f"Final results have {len(final_results)} ranked candidates")

        return StopEvent(result=final_results)


async def main():
    job_description = """
    Job Description:
    We are looking for an experienced and motivated Java Backend Developer to join our dynamic team. The ideal candidate will have a strong background in backend development, particularly with Java, and experience working with Spring Boot, MongoDB, and RESTful APIs. This role requires the ability to develop and maintain complex server-side applications, contribute to system architecture, and collaborate with front-end developers and other stakeholders to deliver high-quality software solutions.
    Key Responsibilities:
    Design, develop, and maintain scalable and efficient backend systems using Java and Spring Boot.
    Work with MongoDB for database management, and integrate RESTful APIs to ensure smooth communication between the backend and front-end applications.
    Collaborate with cross-functional teams to define system requirements, provide technical guidance, and contribute to the development of software architecture.
    Perform unit testing, integration testing, and troubleshooting of backend services.
    Participate in code reviews to ensure best practices and coding standards.
    Use Git, GitHub, and Bitbucket for version control and continuous integration.
    Collaborate with UI/UX teams to ensure the seamless integration of the front-end and backend systems.
    Required Skills:
    Proven experience as a Java Backend Developer with expertise in Spring Boot and MongoDB.
    Strong knowledge of RESTful API design and development.
    Familiarity with Git, GitHub, and Bitbucket for version control and collaborative development.
    Experience with Selenium for automated testing.
    Proficiency in UI/UX testing and ensuring that the backend systems work seamlessly with front-end applications.
    Ability to work effectively in a collaborative team environment.
    Preferred Qualifications:
    Bachelor's degree in Computer Science Engineering or a related field.
    Experience with IntelliJ IDEA as an Integrated Development Environment (IDE).
    Familiarity with front-end frameworks such as PrimeReact is a plus.
    """

    w = JobMatchingWorkflow(timeout=300)
    result = await w.run(job_description=job_description)

    print("Top 5 Matching Candidates:")
    for candidate in result:
        print(f"\nRank: {candidate['rank']}")
        print(f"Name: {candidate['profile'].get('firstName', '')} {candidate['profile'].get('lastName', '')}")
        print(f"Match Score: {candidate['match_score']}")
        print(f"Explanation: {candidate['explanation']}")

if __name__ == "__main__":
    asyncio.run(main())