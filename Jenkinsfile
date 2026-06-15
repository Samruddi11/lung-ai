pipeline {
    agent any

    stages {

        stage('Build Docker Image') {
            steps {
                dir('C:/Users/BHARGAVI/lung-ai/lung-ai') {
                    bat 'docker build -t lung-ai-api .'
                }
            }
        }

        stage('Stop Old Container') {
            steps {
                bat 'docker stop lung-ai-container || exit 0'
                bat 'docker rm lung-ai-container || exit 0'
            }
        }

        stage('Run Container') {
            steps {
                bat 'docker run -d -p 5001:5001 --name lung-ai-container lung-ai-api'
            }
        }
    }
}