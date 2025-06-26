# Urdu Paraphrase Type Identification

A research-based NLP web application for classifying Urdu sentence pairs into 14 different paraphrase types with confidence scoring.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mumarasif/urdu-paraphrase-webapp.git
   cd urdu-paraphrase-webapp
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup database**
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

5. **Access the application**
   Open your browser and go to `http://127.0.0.1:8000`

## Create and Work on a New Branch
1. Before starting any new work always sync your local project with the latest version from GitHub:
   ```bash
   git checkout main
   git pull origin main
   ```
   
2. Always create a new branch for a feature or a task. This avoids conflicts and keeps work separate.

   👉 Example:
   You're working on the classification page:
   ```bash
   git checkout -b feature/classify-page
   ```
   
3. Now you're on the new branch. Make changes in classify.html, test locally, then:
   ```bash
   git add .
   git commit -m "Added frontend layout for classify page"
   ```

4. Push your branch:
   ```bash
   git push origin feature/classify-page
   ```

## Project Structure

```
urdu-paraphrase-webapp/
├── manage.py
├── requirements.txt
├── README.md
├── urdu_paraphrase/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── paraphrase_app/
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   └── views.py
├── templates/
│   ├── base.html
│   └── paraphrase_app/
│       ├── home.html
│       ├── classify.html
│       ├── paraphrase.html
│       └── eda.html
└── scripts/
    └── setup_database.sql
```

## Usage

### 1. Home Page (`/`)
- Introduction to the project
- Overview of features
- Quick navigation to main functions

### 2. Classification Page (`/classify/`)
- Input two Urdu sentences
- Get paraphrase type classification
- View confidence scores and explanations
- Try sample examples

### 3. Paraphrase Generation (`/paraphrase/`)
- Input a single Urdu sentence
- Generate paraphrases automatically
- Classify the generated paraphrase pair
- Regenerate or try new sentences

### 4. Dataset Analysis (`/eda/`)
- View dataset statistics
- Explore class distribution charts
- Analyze sentence length patterns
- Browse sample examples

## Technical Features

- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **RTL Support**: Proper right-to-left text support for Urdu
- **Font Support**: Noto Nastaliq Urdu for authentic Urdu typography
- **Interactive Charts**: Chart.js for data visualization
- **AJAX Requests**: Smooth user experience without page reloads
- **Theme Toggle**: Light and dark mode support
- **Progressive Enhancement**: Works without JavaScript for basic functionality

## API Endpoints

- `POST /api/classify/` - Classify sentence pairs
- `POST /api/paraphrase/` - Generate and classify paraphrases

## Model Integration

The current implementation uses mock responses for demonstration. To integrate with your actual NLP model:

1. Replace the mock functions in `views.py` with your model inference code
2. Add model dependencies to `requirements.txt`
3. Update the API endpoints to handle real predictions

## Contributing

This is a research project for academic purposes. For questions or suggestions, please contact the project maintainer.

## License

Academic use only. Please cite if used in research publications.

---

**Final Year Project – Urdu NLP 2025**
*Developed with ❤️ for Natural Language Processing Research*
