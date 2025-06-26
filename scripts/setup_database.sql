-- Create database tables for the Urdu Paraphrase project
-- This script sets up the basic schema for storing classification results

CREATE TABLE IF NOT EXISTS paraphrase_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL UNIQUE,
    name_urdu VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS classification_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence1 TEXT NOT NULL,
    sentence2 TEXT NOT NULL,
    predicted_type VARCHAR(50),
    confidence_score REAL,
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paraphrase_generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_sentence TEXT NOT NULL,
    generated_paraphrase TEXT NOT NULL,
    predicted_type VARCHAR(50),
    confidence_score REAL,
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert paraphrase types
INSERT OR IGNORE INTO paraphrase_types (name, name_urdu, description) VALUES
('Lexical', 'لغوی', 'Word-level substitutions with synonyms'),
('Syntactic', 'نحوی', 'Changes in sentence structure and word order'),
('Semantic', 'معنوی', 'Meaning-preserving transformations'),
('Morphological', 'صرفی', 'Changes in word forms and morphology'),
('Compound', 'مرکب', 'Combination of multiple paraphrase types'),
('Phrasal', 'فقرہ وار', 'Phrase-level substitutions'),
('Structural', 'ڈھانچہ', 'Overall sentence structure changes'),
('Contextual', 'سیاقی', 'Context-dependent paraphrases'),
('Stylistic', 'اسلوبی', 'Style and register variations'),
('Discourse', 'گفتگو', 'Discourse-level transformations'),
('Pragmatic', 'عملی', 'Pragmatic meaning variations'),
('Temporal', 'وقتی', 'Time-related expression changes'),
('Modal', 'کیفیت', 'Modality and mood changes'),
('Negation', 'نفی', 'Negation-related paraphrases');
