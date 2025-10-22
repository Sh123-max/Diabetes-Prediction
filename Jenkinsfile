pipeline {
    agent any

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/madhav481010/Diabetes-Prediction.git'
            }
        }

        stage('Install Python Dependencies') {
            steps {
                sh 'python3 -m pip install --upgrade pip'
                sh 'python3 -m pip install -r requirements.txt'
            }
        }

        stage('Job 1: Preprocess Data') {
            steps {
                sh 'python3 preprocess.py'
            }
        }

        stage('Job 2: Train and Evaluate Models') {
            steps {
                sh 'python3 train_and_evaluate.py'
            }
        }

        stage('Job 3: Deploy and Predict') {
            steps {
                sh 'python3 deploy.py'
            }
        }

        stage('Start Flask Server') {
            steps {
                sh 'python3 app.py'
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully.'
        }
        failure {
            echo '❌ Pipeline failed.'
        }
    }
}
