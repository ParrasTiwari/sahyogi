from flask import Flask, request, jsonify
import google.generativeai as genai
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

@dataclass
class StudentProfile:
    student_id: str
    name: str
    grade: int
    learning_level: str  # beginner, intermediate, advanced
    learning_style: str  # visual, auditory, kinesthetic, reading_writing
    weaknesses: List[str]  # List of subject areas/topics where student struggles
    strengths: List[str]   # List of subject areas/topics where student excels
    language_preference: str  # Primary language for instruction

@dataclass
class ExerciseRequest:
    student_profile: StudentProfile
    subject: str
    topic: str
    difficulty_level: str
    exercise_count: int
    exercise_types: List[str]  # multiple_choice, short_answer, problem_solving, etc.

class PersonalizedExerciseService:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        
    def generate_exercises(self, exercise_request: ExerciseRequest) -> Dict:
        """
        Generate personalized practice exercises based on student profile and requirements
        """
        try:
            # Create prompt based on student profile and requirements
            prompt = self._create_exercise_prompt(exercise_request)
            
            # Generate exercises using Gemini
            response = self.model.generate_content(prompt)
            
            # Parse and structure the response
            exercises = self._parse_exercises_response(response.text, exercise_request)
            
            return {
                "success": True,
                "student_id": exercise_request.student_profile.student_id,
                "subject": exercise_request.subject,
                "topic": exercise_request.topic,
                "exercises": exercises,
                "generated_at": datetime.now().isoformat(),
                "personalization_factors": {
                    "learning_level": exercise_request.student_profile.learning_level,
                    "learning_style": exercise_request.student_profile.learning_style,
                    "targeted_weaknesses": exercise_request.student_profile.weaknesses
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "student_id": exercise_request.student_profile.student_id
            }
    
    def _create_exercise_prompt(self, req: ExerciseRequest) -> str:
        """
        Create a detailed prompt for Gemini to generate personalized exercises
        """
        student = req.student_profile
        
        prompt = f"""
        You are an expert educational content creator. Generate {req.exercise_count} personalized practice exercises for a student with the following profile:

        STUDENT PROFILE:
        - Name: {student.name}
        - Grade: {student.grade}
        - Learning Level: {student.learning_level}
        - Learning Style: {student.learning_style}
        - Primary Language: {student.language_preference}
        - Weaknesses: {', '.join(student.weaknesses)}
        - Strengths: {', '.join(student.strengths)}

        EXERCISE REQUIREMENTS:
        - Subject: {req.subject}
        - Topic: {req.topic}
        - Difficulty Level: {req.difficulty_level}
        - Exercise Types: {', '.join(req.exercise_types)}

        PERSONALIZATION GUIDELINES:
        1. Target the student's identified weaknesses while building on their strengths
        2. Adapt to their learning style:
           - Visual: Include diagrams, charts, or visual elements
           - Auditory: Include sound-based or rhythm-based elements
           - Kinesthetic: Include hands-on or movement-based activities
           - Reading/Writing: Focus on text-based exercises
        3. Use appropriate complexity for their learning level
        4. If language preference is not English, provide translations or explanations in their preferred language
        5. Create scaffolded exercises that gradually increase in difficulty

        OUTPUT FORMAT:
        Provide the response as a JSON object with the following structure:
        {{
            "exercises": [
                {{
                    "id": 1,
                    "type": "exercise_type",
                    "question": "exercise question/prompt",
                    "options": ["option1", "option2", "option3", "option4"] (for multiple choice),
                    "correct_answer": "correct answer or explanation",
                    "explanation": "detailed explanation of the solution",
                    "difficulty": "beginner/intermediate/advanced",
                    "learning_objective": "what this exercise aims to teach",
                    "personalization_notes": "how this addresses student's specific needs"
                }}
            ]
        }}

        Generate exercises that are engaging, age-appropriate, and specifically designed to help this student overcome their weaknesses while leveraging their learning style.
        """
        
        return prompt
    
    def _parse_exercises_response(self, response_text: str, req: ExerciseRequest) -> List[Dict]:
        """
        Parse the Gemini response and extract structured exercise data
        """
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                return parsed_data.get('exercises', [])
            else:
                # Fallback: create structured exercises from plain text
                return self._create_fallback_exercises(response_text, req)
                
        except json.JSONDecodeError:
            # Fallback parsing if JSON parsing fails
            return self._create_fallback_exercises(response_text, req)
    
    def _create_fallback_exercises(self, text: str, req: ExerciseRequest) -> List[Dict]:
        """
        Create structured exercises when JSON parsing fails
        """
        # Simple fallback - split text into exercises and structure them
        exercises = []
        lines = text.strip().split('\n')
        
        current_exercise = {}
        exercise_count = 0
        
        for line in lines:
            line = line.strip()
            if line and exercise_count < req.exercise_count:
                if not current_exercise:
                    current_exercise = {
                        "id": exercise_count + 1,
                        "type": req.exercise_types[0] if req.exercise_types else "short_answer",
                        "question": line,
                        "correct_answer": "Answer will be provided by teacher",
                        "explanation": "Detailed explanation will be provided",
                        "difficulty": req.difficulty_level,
                        "learning_objective": f"Practice {req.topic} concepts",
                        "personalization_notes": f"Adapted for {req.student_profile.learning_style} learning style"
                    }
                    exercises.append(current_exercise)
                    current_exercise = {}
                    exercise_count += 1
        
        return exercises

# # Flask API endpoints
# app = Flask(__name__)

# @app.route('/generate-exercises', methods=['POST'])
def generate_personalized_exercises(data):
    """
    API endpoint to generate personalized exercises
    
    Expected JSON payload:
    {
        "student_profile": {
            "student_id": "string",
            "name": "string",
            "grade": int,
            "learning_level": "beginner|intermediate|advanced",
            "learning_style": "visual|auditory|kinesthetic|reading_writing",
            "weaknesses": ["topic1", "topic2"],
            "strengths": ["topic1", "topic2"],
            "language_preference": "string"
        },
        "subject": "string",
        "topic": "string",
        "difficulty_level": "string",
        "exercise_count": int,
        "exercise_types": ["multiple_choice", "short_answer", "problem_solving"]
    }
    """
    try:
        # data = request.get_json()
        
        # Validate required fields
        if not data or 'student_profile' not in data:
            return jsonify({"success": False, "error": "Student profile is required"}), 400
        
        # Create student profile
        profile_data = data['student_profile']
        student_profile = StudentProfile(
            student_id=profile_data.get('student_id', ''),
            name=profile_data.get('name', ''),
            grade=profile_data.get('grade', 1),
            learning_level=profile_data.get('learning_level', 'beginner'),
            learning_style=profile_data.get('learning_style', 'visual'),
            weaknesses=profile_data.get('weaknesses', []),
            strengths=profile_data.get('strengths', []),
            language_preference=profile_data.get('language_preference', 'English')
        )
        
        # Create exercise request
        exercise_request = ExerciseRequest(
            student_profile=student_profile,
            subject=data.get('subject', 'Mathematics'),
            topic=data.get('topic', 'Basic Operations'),
            difficulty_level=data.get('difficulty_level', 'beginner'),
            exercise_count=data.get('exercise_count', 5),
            exercise_types=data.get('exercise_types', ['multiple_choice'])
        )
            
        return exercise_request
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({"status": "healthy", "service": "personalized_exercises"})
def generate_synthetic_data():
    student_profile = StudentProfile(
        student_id=1234,
        name='GURU',
        grade=5,
        learning_level='beginner',
        learning_style='visual',
        weaknesses='focus',
        strengths=['visualization','creativity'],
        language_preference='Hindi'
    )
    
    # Create exercise request
    exercise_request = ExerciseRequest(
        student_profile=student_profile,
        subject='Mathematics',
        topic='Basic Operations',
        difficulty_level='beginner',
        exercise_count=5,
        exercise_types=['multiple_choice']
    )
    
    return exercise_request


exercise_service = PersonalizedExerciseService()
if __name__ == '__main__':
    # Set your Gemini API key as an environment variable
    # export GEMINI_API_KEY="your_api_key_here"
    # app.run(host='0.0.0.0', port=5001, debug=True)
    data = generate_synthetic_data()
    exercise_request = generate_personalized_exercises(data)
    result = exercise_service(exercise_request)