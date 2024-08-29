# ai_find_people

- Serviço de monitoramento de streaming de video RTSP usado para identificar se alguma pessoa entrou na area demarcada.

- Requisitos:
    - Python 3.12
    - Processador CPU intel

- Ferramentas utilizadas:
    - Ultralytics
    - YOLO v8 
    - Pytorch
    - OpenVINO (Eficientizar processamento em processadores intel)
    - Mailjet (Enviar notificações por e-mail)

- Como instalar e executar:
    - PIPENV:
        - pipenv install --python 3.12 (Para criar o ambiente virtual pipenv)
        - Defina a area com o criador de area (area_creator.html) e alterando a variável area_of_interest do arquivo find-people.py
        - Defina os parametros para os envios de e-mails e o link RTSP no arquivo .env
        - pipenv shell (Para entrar no ambiente criado)
        - python find-people.py (Para executar o serviço ai_find_people)

- Dificuldades encontradas durante a criação do serviço:
    - Como executar as inferencias do Ultralytics com YOLOv8 no openVINO (para cada frame)
    - Receber um link RTPS e separar frame por frame para executar as inferencias
    - Fazer que as inferencias identifiquem somente pessoas
    - Capturar a imagem com uma pessoa detectada e anexar no e-mail para notificação 
    - Criar uma area para delimitar local de detecção e ignorar os objetos fora dessa area
    - Tratar erro de frame não recebido chamando novamente a função cv2.VideoCapture
    - Quando não se encontra objetos com facilidade no video utilizado -> Testar funcionalidade com outros videos RTSP de teste