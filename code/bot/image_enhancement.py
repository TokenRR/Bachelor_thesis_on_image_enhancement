import numpy as np
import telebot
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array

from tools import change_working_directory_to_current_file
from tools import download_image


def upgrade_image(bot, model, button_message_id, message):
    """
    Функція покращення якості зображення
    """

    change_working_directory_to_current_file()
    try:
        # Завантаження зображення
        LR_image = download_image(bot, message)
        LR_image_help = download_image(bot, message)

        # delete 'Скасувати' button after image is downloaded
        bot.delete_message(chat_id=message.chat.id, message_id=button_message_id)

        # Обробка зображення
        # upscale_factor = 3
        # LR_image = LR_image.resize((LR_image.size[0] // upscale_factor, 
        #                             LR_image.size[1] // upscale_factor), 
        #                             Image.BICUBIC)

        ycbcr = LR_image.convert('YCbCr')
        y, cb, cr = ycbcr.split()
        y = img_to_array(y)
        y = y.astype('float32') / 255.0

        model_input = y.reshape(1, y.shape[0], y.shape[1], y.shape[2])
        output = model.predict(model_input)
        output = output[0]
        output *= 255.0
        output = output.clip(0, 255)
        output = output.reshape((output.shape[0], output.shape[1]))
        output = Image.fromarray(np.uint8(output))
        output = output.resize(LR_image_help.size, Image.Resampling.NEAREST)
        
        cb = cb.resize(output.size, Image.Resampling.BICUBIC)
        cr = cr.resize(output.size, Image.Resampling.BICUBIC)
        ER_image = Image.merge('YCbCr', (output, cb, cr))
        ER_image = ER_image.convert('RGB')
        LR_image = LR_image.resize(ER_image.size, Image.Resampling.BICUBIC)

        # Розрахунок точності
        LR_arr = img_to_array(LR_image)
        ER_arr = img_to_array(ER_image)
        psnr = tf.image.psnr(LR_arr, ER_arr, max_val=255)
        caption = f'PSNR = {psnr:.3f}'

        # Збереження зображень
        LR_image.save('images\\LR_image.jpg')
        ER_image.save('images\\ER_image.jpg')

        # Відправлення зображень
        with open('images\\LR_image.jpg', 'rb') as LR_image, open('images\\ER_image.jpg', 'rb') as ER_image:
            media = [telebot.types.InputMediaPhoto(LR_image, caption=caption),
                        telebot.types.InputMediaPhoto(ER_image),]
            bot.send_media_group(message.chat.id, media, reply_to_message_id=message.message_id)
        
        with open('images\\LR_image.jpg', 'rb') as LR_image, open('images\\ER_image.jpg', 'rb') as ER_image:
            media = [telebot.types.InputMediaDocument(LR_image),
                        telebot.types.InputMediaDocument(ER_image, caption=caption),]
            bot.send_media_group(message.chat.id, media, reply_to_message_id=message.message_id)
    except:
        msg = 'На жаль, не вдалось покращити якість Вашого зображення \nСпробуйте викликати команду /upgrade ще раз'
        bot.send_message(message.chat.id, msg)
