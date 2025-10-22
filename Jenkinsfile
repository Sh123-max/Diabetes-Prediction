pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                echo '📦 Cloning Repository...'
                git branch: 'main', url: 'https://github.com/madhav481010/Diabetes-Prediction.git'
            }
        }

        stage('Setup Python Environment') {
            steps {
                echo '⚙️ Setting up Python Virtual Environment...'
                sh '''
                    # Ensure Python 3 is available
                    python3 --version

                    # Create a virtual environment
                    python3 -m venv venv

                    # Activate the environment and upgrade pip
                    . venv/bin/activate
                    pip install --upgrade pip

                    # Install required dependencies
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Job 1: Preprocess Data') {
            steps {
                echo '🧹 Running Data Preprocessing...'
                sh '''
                    . venv/bin/activate
                    python preprocess.py
                '''
            }
        }

        stage('Job 2: Train and Evaluate Models') {
            steps {
                echo '🤖 Training and Evaluating Models...'
                sh '''
                    . venv/bin/activate
                    python train_and_evaluate.py
                '''
            }
        }

        stage('Job 3: Deploy and Predict') {
            steps {
                echo '🚀 Deploying Model and Running Predictions...'
                sh '''
                    . venv/bin/activate
                    python deploy.py
                '''
            }
        }

        stage('Start Flask Server') {
            steps {
                echo '🌐 Starting Flask Server...'
                sh '''
                    . venv/bin/activate
                    python app.py
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully.'
        }
        failure {
            echo '❌ Pipeline failed. Check the console output for errors.'
        }
        always {
            echo '🧾 Pipeline execution finished.'
        }
    }
}
