import numpy as np
import telebot
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

from tools import change_working_directory_to_current_file
from tools import download_image


def degrade_image(bot, button_message_id, message):
    """
    Функція погіршення якості зображення
    """

    change_working_directory_to_current_file()
    try:
        # Завантаження зображення
        HR_image = download_image(bot, message)

        # delete 'Скасувати' button after image is downloaded
        bot.delete_message(chat_id=message.chat.id, message_id=button_message_id)

        # Зменшення розміру зображення
        small_image = HR_image.resize((HR_image.size[0]//3, HR_image.size[1]//3), Image.Resampling.BICUBIC)

        # Збільшення розміру зображення назад до оригінального розміру
        LR_image = small_image.resize(HR_image.size, Image.Resampling.BICUBIC)

        # Збереження зображень
        HR_image.save('images\\HR_image.jpg')
        LR_image.save('images\\LR_image.jpg')

        # Конвертація зображень в масиви NumPy
        HR_image = np.array(HR_image)
        LR_image = np.array(LR_image)

        psnr = peak_signal_noise_ratio(HR_image, LR_image)

        # Відправлення зображень
        caption = f'PSNR = {psnr:.3f}'
        with open('images\\HR_image.jpg', 'rb') as HR_image, open('images\\LR_image.jpg', 'rb') as LR_image:
            media = [telebot.types.InputMediaPhoto(LR_image),
                        telebot.types.InputMediaPhoto(HR_image, caption=caption),]
            bot.send_media_group(message.chat.id, media, reply_to_message_id=message.message_id)

        with open('images\\HR_image.jpg', 'rb') as HR_image, open('images\\LR_image.jpg', 'rb') as LR_image:
            media = [telebot.types.InputMediaDocument(LR_image),
                        telebot.types.InputMediaDocument(HR_image, caption=caption),]
            bot.send_media_group(message.chat.id, media, reply_to_message_id=message.message_id)
    except Exception as e:
        msg = 'На жаль, не вдалось погіршити якість Вашого зображення \nСпробуйте викликати команду /degrade ще раз'
        bot.send_message(message.chat.id, msg)
