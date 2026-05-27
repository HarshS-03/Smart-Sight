# Smart Sight

A Django-based face recognition and detection system for managing and analyzing facial data.

## Features

- Face detection and recognition
- Person dataset management
- Recognition logging and reports
- Admin dashboard for monitoring
- Support for multiple detection models (nano, small)

## Requirements

- Python 3.8+
- Django 6.0+
- OpenCV
- PyTorch
- See `requirements.txt` for full dependencies

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Smart\ Sight
   ```
2. **Create and activate virtual environment**

   ```bash
   python -m venv env
   env\Scripts\activate  # On Windows
   source env/bin/activate  # On macOS/Linux
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run migrations**

   ```bash
   python manage.py migrate
   ```
5. **Create superuser**

   ```bash
   python manage.py createsuperuser
   ```
6. **Start development server**

   ```bash
   python manage.py runserver
   ```

## Usage

- Access admin panel at `http://localhost:8000/admin`
- Upload person datasets in media/dataset directory
- Run detection/recognition on images
- View reports and logs in dashboard

## Project Structure

```
Smart Sight/
├── app/                 # Main application
│   ├── models/         # Detection models (nano, small)
│   ├── templates/      # HTML templates
│   ├── static/         # Static files
│   └── migrations/     # Database migrations
├── smartsight/         # Django project settings
├── media/              # User uploaded datasets
└── manage.py          # Django management script
```
