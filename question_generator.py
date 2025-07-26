#!/usr/bin/env python3
"""
Question Paper Generator using google/flan-t5-large model
"""

import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict, Optional
import json

class QuestionPaperGenerator:
    def __init__(self, model_path: str = "./model_cache", device: str = "auto"):
        """
        Initialize the Question Paper Generator
        
        Args:
            model_path: Path to the cached model
            device: Device to run the model on ("auto", "cpu", "cuda")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the flan-t5-large model and tokenizer"""
        try:
            print("🔄 Loading flan-t5-large model...")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(
                "google/flan-t5-large",
                cache_dir=self.model_path
            )
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-large",
                cache_dir=self.model_path,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                device_map="cpu"  # Use CPU to avoid MPS issues
            )
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure you've run download_model.py first!")
    
    def generate_questions(self, 
                          text: str, 
                          question_type: str = "mcq",
                          num_questions: int = 5,
                          difficulty: str = "medium",
                          temperature: float = 0.7) -> str:
        """
        Generate questions based on the provided text
        
        Args:
            text: The source text to generate questions from
            question_type: Type of questions ("mcq", "short_answer", "essay", "true_false")
            num_questions: Number of questions to generate
            difficulty: Difficulty level ("easy", "medium", "hard")
            temperature: Controls randomness (0.1-1.0)
        
        Returns:
            Generated questions as a string
        """
        if not self.model or not self.tokenizer:
            return "❌ Model not loaded. Please check the model installation."
        
        # Create appropriate prompt based on question type
        prompt = self._create_prompt(text, question_type, num_questions, difficulty)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=800,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            return f"❌ Error generating questions: {e}"
    
    def _create_prompt(self, text: str, question_type: str, num_questions: int, difficulty: str) -> str:
        """Create a structured prompt for question generation"""
        
        question_type_instructions = {
            "mcq": f"Generate {num_questions} multiple choice questions with 4 options (A, B, C, D) each. Include the correct answer marked as 'Correct Answer:'",
            "short_answer": f"Generate {num_questions} short answer questions. Provide sample answers for each question.",
            "essay": f"Generate {num_questions} essay questions. Provide a brief outline of expected points for each question.",
            "true_false": f"Generate {num_questions} true/false questions. Mark the correct answer for each.",
            "fill_blank": f"Generate {num_questions} fill-in-the-blank questions. Provide the correct answers."
        }
        
        difficulty_instruction = f"Make the questions {difficulty} difficulty level."
        
        prompt = f"""Given the following text, {question_type_instructions.get(question_type, question_type_instructions['mcq'])} {difficulty_instruction}

Text:
{text}

Questions:"""
        
        return prompt
    
    def generate_question_paper(self, 
                               text: str,
                               paper_config: Dict) -> Dict:
        """
        Generate a complete question paper with multiple sections
        
        Args:
            text: Source text/material
            paper_config: Configuration for the paper
                {
                    "title": "Paper Title",
                    "total_marks": 100,
                    "time_allowed": "3 hours",
                    "sections": [
                        {
                            "name": "Section A",
                            "type": "mcq",
                            "num_questions": 10,
                            "marks_per_question": 2,
                            "difficulty": "easy"
                        },
                        {
                            "name": "Section B", 
                            "type": "short_answer",
                            "num_questions": 5,
                            "marks_per_question": 8,
                            "difficulty": "medium"
                        }
                    ]
                }
        
        Returns:
            Complete question paper as a dictionary
        """
        question_paper = {
            "title": paper_config.get("title", "Question Paper"),
            "total_marks": paper_config.get("total_marks", 100),
            "time_allowed": paper_config.get("time_allowed", "3 hours"),
            "sections": []
        }
        
        for section in paper_config.get("sections", []):
            section_questions = self.generate_questions(
                text=text,
                question_type=section["type"],
                num_questions=section["num_questions"],
                difficulty=section["difficulty"]
            )
            
            question_paper["sections"].append({
                "name": section["name"],
                "type": section["type"],
                "marks_per_question": section["marks_per_question"],
                "questions": section_questions
            })
        
        return question_paper
    
    def save_question_paper(self, question_paper: Dict, filename: str = "question_paper.json"):
        """Save the generated question paper to a JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(question_paper, f, indent=2, ensure_ascii=False)
            print(f"✅ Question paper saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving question paper: {e}")

def main():
    """Example usage of the Question Paper Generator"""
    
    # Initialize the generator
    generator = QuestionPaperGenerator()
    
    # Sample text for question generation
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that can perform tasks that typically require human intelligence. These tasks include learning, 
    reasoning, problem-solving, perception, and language understanding. Machine learning is a subset 
    of AI that enables computers to learn and improve from experience without being explicitly programmed. 
    Deep learning, a subset of machine learning, uses neural networks with multiple layers to analyze 
    various factors of data. AI applications include virtual assistants, recommendation systems, 
    autonomous vehicles, and medical diagnosis systems.
    """
    
    # Example 1: Generate MCQ questions
    print("📝 Generating MCQ Questions...")
    mcq_questions = generator.generate_questions(
        text=sample_text,
        question_type="mcq",
        num_questions=3,
        difficulty="medium"
    )
    print(mcq_questions)
    print("\n" + "="*50 + "\n")
    
    # Example 2: Generate a complete question paper
    print("📄 Generating Complete Question Paper...")
    paper_config = {
        "title": "Artificial Intelligence - Mid Term Examination",
        "total_marks": 50,
        "time_allowed": "1 hour",
        "sections": [
            {
                "name": "Section A: Multiple Choice Questions",
                "type": "mcq",
                "num_questions": 5,
                "marks_per_question": 2,
                "difficulty": "easy"
            },
            {
                "name": "Section B: Short Answer Questions", 
                "type": "short_answer",
                "num_questions": 3,
                "marks_per_question": 8,
                "difficulty": "medium"
            },
            {
                "name": "Section C: Essay Question",
                "type": "essay", 
                "num_questions": 1,
                "marks_per_question": 16,
                "difficulty": "hard"
            }
        ]
    }
    
    question_paper = generator.generate_question_paper(sample_text, paper_config)
    
    # Save the question paper
    generator.save_question_paper(question_paper, "ai_question_paper.json")
    
    # Print the question paper
    print("\n📋 Generated Question Paper:")
    print(json.dumps(question_paper, indent=2))

if __name__ == "__main__":
    main() 