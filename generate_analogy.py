import google.generativeai as genai
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import re
# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

@dataclass
class StudentContext:
    student_id: str
    name: str
    grade: int
    region: str  # Geographic region/state/country
    local_language: str  # Primary local language
    cultural_context: str  # Rural/Urban/Tribal etc.
    familiar_concepts: List[str]  # Local concepts student is familiar with

@dataclass
class AnalogyRequest:
    student_context: StudentContext
    question: str  # Student's question in local language or English
    subject: str
    topic: str
    complexity_level: str  # simple, moderate, complex
    preferred_language: str  # Language for response

class LocalAnalogyService:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Predefined cultural and regional contexts for better analogies
        self.regional_contexts = {
            "rural_india": {
                "common_concepts": ["farming", "village market", "bullock cart", "monsoon", "harvest", "cattle", "well", "temple"],
                "occupations": ["farmer", "shopkeeper", "teacher", "blacksmith", "weaver"],
                "festivals": ["Diwali", "Holi", "harvest festival"],
                "food": ["rice", "wheat", "dal", "curry", "chapati"]
            },
            "urban_india": {
                "common_concepts": ["traffic", "metro", "mall", "apartment", "office", "smartphone", "internet"],
                "occupations": ["engineer", "doctor", "business person", "teacher", "driver"],
                "festivals": ["Diwali", "Christmas", "New Year"],
                "food": ["street food", "restaurant", "fast food", "traditional meals"]
            },
            "tribal_areas": {
                "common_concepts": ["forest", "tribal dance", "traditional crafts", "nature", "community gathering"],
                "occupations": ["hunter", "gatherer", "craft maker", "traditional healer"],
                "festivals": ["harvest ceremonies", "tribal festivals"],
                "food": ["forest produce", "traditional recipes", "wild vegetables"]
            }
        }
    
    def generate_analogy(self, analogy_request: AnalogyRequest) -> Dict:
        """
        Generate culturally relevant analogies for complex concepts
        """
        try:
            # Detect and translate the question if needed
            detected_language, translated_question = self._process_question(
                analogy_request.question, 
                analogy_request.preferred_language
            )
            
            # Generate culturally relevant analogy
            analogy_response = self._create_analogy(analogy_request, translated_question)
            
            # Translate response back to preferred language if needed
            final_response = self._translate_response(
                analogy_response, 
                analogy_request.preferred_language,
                detected_language
            )
            
            return {
                "success": True,
                "student_id": analogy_request.student_context.student_id,
                "original_question": analogy_request.question,
                "detected_language": detected_language,
                "translated_question": translated_question,
                "subject": analogy_request.subject,
                "topic": analogy_request.topic,
                "analogy": final_response,
                "cultural_context": analogy_request.student_context.cultural_context,
                "region": analogy_request.student_context.region,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "student_id": analogy_request.student_context.student_id
            }
    
    def _process_question(self, question: str, preferred_language: str) -> tuple:
        """
        Detect language and translate question to English for processing
        """
        try:
            # Language detection prompt
            detection_prompt = f"""
            Analyze the following text and determine the language it's written in:
            "{question}"
            
            Respond with just the language name (e.g., "Hindi", "Bengali", "Tamil", "English", etc.)
            """
            
            detection_response = self.model.generate_content(detection_prompt)
            detected_language = detection_response.text.strip()
            
            # If not English, translate to English
            if detected_language.lower() != "english":
                translation_prompt = f"""
                Translate the following {detected_language} text to English:
                "{question}"
                
                Provide only the English translation without any additional text.
                """
                
                translation_response = self.model.generate_content(translation_prompt)
                translated_question = translation_response.text.strip()
            else:
                translated_question = question
            
            return detected_language, translated_question
            
        except Exception as e:
            # Fallback: assume English
            return "English", question
    
    def _create_analogy(self, req: AnalogyRequest, translated_question: str) -> str:
        """
        Create culturally relevant analogy based on student's context
        """
        context = req.student_context
        
        # Get regional context
        cultural_key = self._get_cultural_key(context.cultural_context, context.region)
        regional_info = self.regional_contexts.get(cultural_key, self.regional_contexts["rural_india"])
        
        analogy_prompt = f"""
        You are an expert teacher who specializes in creating culturally relevant analogies for students. 
        
        STUDENT CONTEXT:
        - Name: {context.name}
        - Grade: {context.grade}
        - Region: {context.region}
        - Local Language: {context.local_language}
        - Cultural Context: {context.cultural_context}
        - Familiar Concepts: {', '.join(context.familiar_concepts)}
        
        REGIONAL/CULTURAL INFORMATION:
        - Common local concepts: {', '.join(regional_info['common_concepts'])}
        - Local occupations: {', '.join(regional_info['occupations'])}
        - Local festivals: {', '.join(regional_info['festivals'])}
        - Local food: {', '.join(regional_info['food'])}
        
        QUESTION TO EXPLAIN: "{translated_question}"
        SUBJECT: {req.subject}
        TOPIC: {req.topic}
        COMPLEXITY LEVEL: {req.complexity_level}
        
        TASK:
        Create a detailed, culturally relevant analogy that explains the concept in the question using:
        1. Local and familiar concepts from the student's environment
        2. Everyday experiences the student can relate to
        3. Simple language appropriate for grade {context.grade}
        4. Step-by-step explanation building from familiar to unfamiliar
        5. Examples from local culture, geography, or daily life
        
        STRUCTURE YOUR RESPONSE AS:
        1. **Simple Explanation**: Start with a basic explanation in simple terms
        2. **Local Analogy**: Provide a detailed analogy using familiar local concepts
        3. **Connection**: Clearly connect the analogy back to the original concept
        4. **Example**: Give a practical example from the student's local environment
        5. **Summary**: Summarize the key learning points
        
        Make sure the analogy is:
        - Culturally sensitive and appropriate
        - Age-appropriate for grade {context.grade}
        - Easy to understand and remember
        - Relevant to the student's daily life experience
        """
        
        response = self.model.generate_content(analogy_prompt)
        return response.text
    
    def _get_cultural_key(self, cultural_context: str, region: str) -> str:
        """
        Determine the appropriate cultural context key
        """
        cultural_lower = cultural_context.lower()
        region_lower = region.lower()
        
        if "tribal" in cultural_lower or "adivasi" in cultural_lower:
            return "tribal_areas"
        elif "urban" in cultural_lower or "city" in cultural_lower:
            return "urban_india"
        else:
            return "rural_india"
    
    def _translate_response(self, response: str, preferred_language: str, detected_language: str) -> str:
        """
        Translate the analogy response to the preferred language if needed
        """
        try:
            if preferred_language.lower() != "english" and detected_language.lower() != "english":
                translation_prompt = f"""
                Translate the following English text to {preferred_language}, maintaining the structure and cultural references:
                
                "{response}"
                
                Make sure to:
                1. Keep the educational structure intact
                2. Maintain cultural references and analogies
                3. Use appropriate language level for a grade school student
                4. Preserve the key learning concepts
                
                Provide only the translated text.
                """
                
                translation_response = self.model.generate_content(translation_prompt)
                return translation_response.text.strip()
            
            return response
            
        except Exception as e:
            # Fallback: return original response
            return response
    
    def get_supported_languages(self) -> List[str]:
        """
        Return list of supported languages for analogies
        """
        return [
            "English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", 
            "Gujarati", "Kannada", "Malayalam", "Punjabi", "Odia", "Assamese"
        ]

def generate_local_analogy(data):
    """
    API endpoint to generate culturally relevant analogies
    
    Expected JSON payload:
    {
        "student_context": {
            "student_id": "string",
            "name": "string", 
            "grade": int,
            "region": "string",
            "local_language": "string",
            "cultural_context": "rural|urban|tribal",
            "familiar_concepts": ["concept1", "concept2"]
        },
        "question": "string (in any supported language)",
        "subject": "string",
        "topic": "string", 
        "complexity_level": "simple|moderate|complex",
        "preferred_language": "string"
    }
    """
     
    # Validate required fields
    if not data or 'student_context' not in data or 'question' not in data:
        return {
            "success": False, 
            "error": "Student context and question are required"
        }
    
    # Create student context
    context_data = data['student_context']
    student_context = StudentContext(
        student_id=context_data.get('student_id', ''),
        name=context_data.get('name', ''),
        grade=context_data.get('grade', 1),
        region=context_data.get('region', 'Rural India'),
        local_language=context_data.get('local_language', 'Hindi'),
        cultural_context=context_data.get('cultural_context', 'rural'),
        familiar_concepts=context_data.get('familiar_concepts', [])
    )
    
    # Create analogy request
    analogy_request = AnalogyRequest(
        student_context=student_context,
        question=data.get('question', ''),
        subject=data.get('subject', 'General'),
        topic=data.get('topic', 'Basic Concepts'),
        complexity_level=data.get('complexity_level', 'simple'),
        preferred_language=data.get('preferred_language', 'English')
    )
    
    return analogy_request

if __name__ == '__main__':
    # Set your Gemini API key as an environment variable
    # export GEMINI_API_KEY="your_api_key_here"
    data = {
        "student_context": {
            "student_id": "123",
            "name": "GURU", 
            "grade": 5,
            "region": "north india",
            "local_language": "hindi",
            "cultural_context": "rural",
            "familiar_concepts": ["farming",]
        },
        "question": "ye barish kyu hoti hai?",
        "subject": "environment",
        "topic": "string", 
        "complexity_level": "simple",
        "preferred_language": "hindi"
    }

    analogy_request = generate_local_analogy(data)
    analogy_service = LocalAnalogyService()
    result = analogy_service.generate_analogy(analogy_request)
    print(result)
    
