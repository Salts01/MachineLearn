import os
import tensorflow as tf

base_dir = 'C:/Users/User/Python/MachineLearn/ExpressionIdentification/Data'
ext_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

erros = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        caminho = os.path.join(root, file)
        if file.lower().endswith(ext_validas):
            try:
                img_raw = tf.io.read_file(caminho)
                img = tf.image.decode_image(img_raw)
            except Exception as e:
                print(f"❌ Erro ao abrir: {caminho}")
                print(f"↳ {e}")
                erros.append(caminho)

if not erros:
    print("✅ Todas as imagens são válidas!")
else:
    print(f"\nForam encontradas {len(erros)} imagens com erro.")