import json
import time
import datetime
from typing_extensions import TypedDict

class AgencyGenerator:
    def __init__(self):
        pass

    def load_data_from_json(self, businesses_file, profiles_file, case_studies_file):
        """Load businesses, profiles, and case studies from JSON files"""
        businesses = []
        profiles = []
        case_studies = []

        try:
            with open(businesses_file, 'r', encoding='utf-8') as f:
                businesses = json.load(f)
            print(f"Loaded {len(businesses)} businesses from {businesses_file}")
        except Exception as e:
            print(f"Error loading businesses from {businesses_file}: {e}")

        try:
            with open(profiles_file, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
            print(f"Loaded {len(profiles)} profiles from {profiles_file}")
        except Exception as e:
            print(f"Error loading profiles from {profiles_file}: {e}")

        try:
            with open(case_studies_file, 'r', encoding='utf-8') as f:
                case_studies = json.load(f)
            print(f"Loaded {len(case_studies)} case studies from {case_studies_file}")
        except Exception as e:
            print(f"Error loading case studies from {case_studies_file}: {e}")

        return businesses, profiles, case_studies

    def extract_profile_skills(self, profile):
        """Extract unique skills from a profile"""
        unique_skills = set()

        highlighted_skills = profile.get('highlightedSkills', [])
        for skill in highlighted_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    unique_skills.add(skill_name)
            elif isinstance(skill, str):
                unique_skills.add(skill)

        additional_skills = profile.get('additionalSkill', [])
        for skill in additional_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    unique_skills.add(skill_name)
            elif isinstance(skill, str):
                unique_skills.add(skill)

        experience_entries = profile.get('experience', [])
        for exp in experience_entries:
            if isinstance(exp, dict):
                practiced_skills = exp.get('practicedSkills', [])
                for skill in practiced_skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', '')
                        if skill_name:
                            unique_skills.add(skill_name)
                    elif isinstance(skill, str):
                        unique_skills.add(skill)

        return list(unique_skills)

    def extract_case_study_skills(self, case_study):
        """Extract unique skills from a case study"""
        unique_skills = set()

        tech_stack = case_study.get('techStack', [])
        for tech in tech_stack:
            if isinstance(tech, dict):
                tech_name = tech.get('name', '')
                if tech_name:
                    unique_skills.add(tech_name)
            elif isinstance(tech, str):
                unique_skills.add(tech)

        teams = case_study.get('teams', [])
        for team_member in teams:
            if isinstance(team_member, dict):
                member_skills = team_member.get('highlightedSkills', [])
                for skill in member_skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', '')
                        if skill_name:
                            unique_skills.add(skill_name)
                    elif isinstance(skill, str):
                        unique_skills.add(skill)

        return list(unique_skills)

    def get_freelancer_name(self, profile):
        """Get the freelancer's full name from the profile document"""
        first_name = profile.get('firstName', '')
        last_name = profile.get('lastName', '')

        if first_name or last_name:
            return f"{first_name} {last_name}".strip()

        full_name = profile.get('fullName', '')
        if full_name:
            return full_name

        name = profile.get('name', '')
        if name:
            return name

        return "Unknown Freelancer"

    def generate_representative_text(self, business, profiles_list, case_studies_list):
        """Generate representative text based on the business, profiles and its case studies"""
        business_name = business.get('name', 'Unknown Business')

        if not profiles_list:
            return None

        is_freelancer = len(profiles_list) == 1
        entity_type = "freelancer" if is_freelancer else "agency"

        if is_freelancer:
            profile = profiles_list[0]
            entity_name = self.get_freelancer_name(profile)
        else:
            entity_name = business_name

        all_profile_skills = []
        for profile in profiles_list:
            all_profile_skills.extend(self.extract_profile_skills(profile))
        unique_profile_skills = list(dict.fromkeys(all_profile_skills))

        case_studies_data = []
        for case_study in case_studies_list:
            industry = case_study.get('industry', 'Unknown Industry')
            skills = self.extract_case_study_skills(case_study)
            if skills and industry:
                case_studies_data.append((industry, skills))

        if not unique_profile_skills:
            rep_text = f"{entity_name} is a {entity_type}"
        else:
            rep_text = f"{entity_name} is a {entity_type} with skills in {', '.join(unique_profile_skills)}"

        if case_studies_data:
            if is_freelancer:
                rep_text += ". They have done projects on "
            else:
                rep_text += ". They have worked on "
            rep_text += " ; ".join([f"{industry} using {', '.join(skills)}" for industry, skills in case_studies_data])

        return rep_text

    def process_businesses(self, businesses, profiles, case_studies):
        """Process businesses to generate representative text for each"""
        processed_docs = []

        profiles_by_business_id = {}
        for profile in profiles:
            business_id = profile.get('businessId', '')
            if business_id:
                if isinstance(business_id, dict) and '$oid' in business_id:
                    business_id = business_id.get('$oid', '')

                if business_id not in profiles_by_business_id:
                    profiles_by_business_id[business_id] = []
                profiles_by_business_id[business_id].append(profile)

        case_studies_by_business_id = {}
        for case_study in case_studies:
            business_id = case_study.get('caseStudyBusinessId', '')
            if business_id:
                if isinstance(business_id, dict) and '$oid' in business_id:
                    business_id = business_id.get('$oid', '')

                if business_id not in case_studies_by_business_id:
                    case_studies_by_business_id[business_id] = []
                case_studies_by_business_id[business_id].append(case_study)

        for business in businesses:
            processed_doc = business.copy()

            business_id = business.get('_id', '')
            if isinstance(business_id, dict) and '$oid' in business_id:
                business_id = business_id.get('$oid', '')

            business_profiles = profiles_by_business_id.get(business_id, [])
            business_case_studies = case_studies_by_business_id.get(business_id, [])

            if business_profiles:
                rep_text = self.generate_representative_text(business, business_profiles, business_case_studies)
                if rep_text:
                    processed_doc["representativeText"] = rep_text

            processed_docs.append(processed_doc)

        return processed_docs

    def save_processed_documents(self, processed_docs, output_file=None):
        """Save processed documents to a JSON file"""
        if not output_file:
            timestamp = int(time.time())
            output_file = f"processed_businesses_{timestamp}.json"

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_docs, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved {len(processed_docs)} documents to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving processed documents: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to handle the generation of representative text"""
    try:
        print("Representative Text Generator for Businesses")
        print("===========================================")

        print("Loading data from JSON files...")

        businesses_file = input("Enter path to businesses JSON file: ").strip()
        profiles_file = input("Enter path to profiles JSON file: ").strip()
        case_studies_file = input("Enter path to case studies JSON file: ").strip()

        generator = AgencyGenerator()
        businesses, profiles, case_studies = generator.load_data_from_json(
            businesses_file, profiles_file, case_studies_file
        )

        if not businesses:
            print("No businesses found. Exiting.")
            return

        print(f"Starting to process {len(businesses)} businesses...")
        start_time = time.time()

        processed_docs = generator.process_businesses(businesses, profiles, case_studies)

        if processed_docs:
            output_file = input("Enter output file name (leave empty for default): ").strip()
            if not output_file:
                timestamp = int(time.time())
                output_file = f"processed_businesses_{timestamp}.json"

            saved_file = generator.save_processed_documents(processed_docs, output_file)

            end_time = time.time()

            if saved_file:
                print(f"\nSummary:")
                print(f"Successfully processed and saved {len(processed_docs)} documents to {saved_file}")
                print(f"Processing completed in {end_time - start_time:.2f} seconds")

                with_rep_text = sum(1 for doc in processed_docs if "representativeText" in doc)
                without_rep_text = len(processed_docs) - with_rep_text
                print(f"Businesses with representative text: {with_rep_text}")
                print(f"Businesses without representative text (no profiles): {without_rep_text}")

                print("\n--- Sample Processed Entry ---")
                if processed_docs:
                    sample = next((doc for doc in processed_docs if "representativeText" in doc), processed_docs[0])
                    print(json.dumps(sample, indent=2))
        else:
            print("No documents were processed.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()