pipeline {
  environment {
    registry = "registry.ifora.hse.ru/microservice_indicators"
    registryCredential = '2347b9bd-46aa-42ad-b775-758e40e83d80'
    dockerImageGit = ''
    dockerImageLatest = ''
    GIT_TAG = "${BRANCH_NAME}"
    def scannerHome = tool 'sonarqube-scanner-4.8.1.3023'
    shortCommit = sh(returnStdout: true, script: "git log -n 1 --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit")
  }

  agent any
  stages {
    stage("Анализ кода при помощи sonarqube") {
      steps {
         withSonarQubeEnv('sonarqube-server') {
         sh '${scannerHome}/bin/sonar-scanner'
        }
      }
    }

    stage("Проверка результата анализа кода sonarqube") {
      steps {
       catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE')
       {
        timeout(time: 1, unit: 'HOURS') {
        waitForQualityGate abortPipeline: true
        }
       }
     }
    }

    stage('Установка зависимостей pdm') {
     steps {
       sh 'pdm install'
     }
    }

    stage('Запуск тестов pdm') {
     steps {
       sh 'pdm run tox'
     }
    }

    stage('Сборка docker образа для prod') {
      when{buildingTag()}
      steps{
        script {
          dockerImageLatest = docker.build registry + ":prod"
          dockerImageGit = docker.build registry + ":$GIT_TAG"
        }
      }
    }

    stage ('Подтвердить деплой на prod') {
      when{buildingTag()}
      steps{
            script {
              timeout(time: 15, unit: 'MINUTES') {
                input(message: "Задеплоить на prod?", ok: 'Deploy')
              }
            }
          }
        }

    stage('Проверка сделан ли тэг из ветки master') {
      when {
        buildingTag()
      }
      steps {
        git branch: 'master',
          credentialsId: '68f3d28f-ce5a-42d4-9ec4-9f6973c841d8',
          url: "${GIT_URL}"
        script {
          sh 'git checkout -f "${GIT_COMMIT}"'
          def CHECK_BRANCH = sh(returnStdout: true, script: 'git branch master --contains "${GIT_COMMIT}"').trim()
          echo "${CHECK_BRANCH}"
          if (CHECK_BRANCH == 'master') {
            echo 'Тэг из ветки master'
          } else {
            echo 'Тэг не из ветки master'
            sh 'exit 1'
          }
        }
      }
    }

    stage('Push docker образа в registry') {
      when{buildingTag()}
      steps{
        script {
          docker.withRegistry('https://registry.ifora.hse.ru', registryCredential ) {
            dockerImageLatest.push()
            dockerImageGit.push()
          }
        }
      }
    }


    stage ('Деплой на prod службы api_indicators_v1') {
      when{buildingTag()}
      steps{
        sshagent(credentials : ["25c5446e-05fc-400c-a7ad-e9bd4b3a26be"]) {
            sh 'ssh -o StrictHostKeyChecking=no jenkinsaccess@172.18.207.29 uptime'
              sh 'ssh -v jenkinsaccess@172.18.207.29 "docker service update --force --with-registry-auth --image registry.ifora.hse.ru/microservice_indicators:prod api_indicators"'
     }
    }
   }


  }

     post {
       success {
            mattermostSend (
              color: "#00f514",
              message: "Build SUCCESS: ${env.JOB_NAME} #${env.BUILD_NUMBER} (<${env.BUILD_URL}|Link to build>) ${env.shortCommit}"
              )
        }

       failure {
            mattermostSend (
              color: "#e00707",
              message: "Build FAILED: ${env.JOB_NAME} #${env.BUILD_NUMBER} (<${env.BUILD_URL}|Link to build>) ${env.shortCommit}"
              )
        }
    }
}
