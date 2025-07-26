# Question Paper Generator

A Python-based question paper generator using the `google/flan-t5-large` model from Hugging Face. This tool can generate various types of questions (MCQ, short answer, essay, true/false) based on provided text content.

## Features

- 🤖 Uses Google's Flan-T5-Large model for intelligent question generation
- 📝 Supports multiple question types: MCQ, Short Answer, Essay, True/False, Fill-in-the-blank
- 🎯 Configurable difficulty levels (Easy, Medium, Hard)
- 📊 Generate complete question papers with multiple sections
- 💾 Save generated papers in JSON format
- ⚡ Optimized for memory efficiency with half-precision (float16)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Model

Run the download script to get the `google/flan-t5-large` model:

```bash
python download_model.py
```

This will:
- Download the model (~3GB) to `./model_cache/`
- Test the model with a sample prompt
- Verify everything is working correctly

## Usage

### Quick Start

```python
from question_generator import QuestionPaperGenerator

# Initialize the generator
generator = QuestionPaperGenerator()

# Generate MCQ questions
text = "Your educational content here..."
questions = generator.generate_questions(
    text=text,
    question_type="mcq",
    num_questions=5,
    difficulty="medium"
)
print(questions)
```

### Generate Complete Question Paper

```python
# Define paper configuration
paper_config = {
    "title": "Computer Science - Final Examination",
    "total_marks": 100,
    "time_allowed": "3 hours",
    "sections": [
        {
            "name": "Section A: Multiple Choice",
            "type": "mcq",
            "num_questions": 10,
            "marks_per_question": 2,
            "difficulty": "easy"
        },
        {
            "name": "Section B: Short Answer",
            "type": "short_answer", 
            "num_questions": 5,
            "marks_per_question": 8,
            "difficulty": "medium"
        },
        {
            "name": "Section C: Essay",
            "type": "essay",
            "num_questions": 2,
            "marks_per_question": 20,
            "difficulty": "hard"
        }
    ]
}

# Generate the paper
question_paper = generator.generate_question_paper(text, paper_config)

# Save to file
generator.save_question_paper(question_paper, "my_question_paper.json")
```

### Run Example

```bash
python question_generator.py
```

This will generate a sample question paper based on AI content and save it as `ai_question_paper.json`.

## Question Types Supported

| Type | Description | Example Output |
|------|-------------|----------------|
| `mcq` | Multiple Choice Questions | Questions with A, B, C, D options |
| `short_answer` | Short Answer Questions | Questions with sample answers |
| `essay` | Essay Questions | Questions with answer outlines |
| `true_false` | True/False Questions | Binary choice questions |
| `fill_blank` | Fill-in-the-blank | Questions with blanks to fill |

## Configuration Options

### Generation Parameters

- `temperature` (0.1-1.0): Controls randomness (higher = more creative)
- `max_length`: Maximum length of generated response
- `num_questions`: Number of questions to generate
- `difficulty`: Easy, Medium, or Hard

### Model Settings

- `device`: "auto", "cpu", or "cuda" for GPU acceleration
- `torch_dtype`: "float16" for memory efficiency, "float32" for precision

## File Structure

```
question-paper-generator/
├── requirements.txt          # Python dependencies
├── download_model.py         # Model download script
├── question_generator.py     # Main generator class
├── README.md                # This file
├── model_cache/             # Downloaded model files
└── *.json                   # Generated question papers
```

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: ~4GB for model cache
- **GPU**: Optional but recommended for faster generation

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Use `torch.float16` (already configured)
   - Reduce batch size or max_length
   - Close other applications

2. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space
   - Try running `download_model.py` again

3. **Slow Generation**
   - Use GPU if available
   - Reduce temperature for faster, more focused generation
   - Use smaller max_length values

### Performance Tips

- Use GPU acceleration when available
- Keep temperature between 0.5-0.8 for balanced creativity
- Use appropriate max_length based on question type
- Cache the model for faster subsequent runs

## Example Output

```json
{
  "title": "Artificial Intelligence - Mid Term Examination",
  "total_marks": 50,
  "time_allowed": "1 hour",
  "sections": [
    {
      "name": "Section A: Multiple Choice Questions",
      "type": "mcq",
      "marks_per_question": 2,
      "questions": "1. What is the primary goal of Artificial Intelligence?\nA) To replace humans\nB) To create intelligent machines\nC) To reduce costs\nD) To increase speed\nCorrect Answer: B\n\n2. Which subset of AI enables computers to learn from experience?\nA) Deep learning\nB) Machine learning\nC) Neural networks\nD) Expert systems\nCorrect Answer: B"
    }
  ]
}
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - see LICENSE file for details. 