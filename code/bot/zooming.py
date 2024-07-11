import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import telebot
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.losses import MeanSquaredError, MeanAbsoluteError
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from tools import download_image
from tools import change_working_directory_to_current_file


def plot_results(img, prefix, title, z1=200, z2=300, z3=150, z4=250):
    """
    Функція для побудови графіків зображень і збереження їх у файл.
    
    Параметри:
    img (np.array): Вхідне зображення.
    prefix (str): Префікс для імені файлу зображення.
    title (str): Заголовок графіка.
    z1, z2, z3, z4 (int, optional): Координати для області збільшення. За замовчуванням 200, 300, 150, 250.
    """

    # Зображення готуються до побудови графіків шляхом перетворення їх у масиви та масштабування їхніх значень
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0

    # Для вихідного зображення та збільшеного зображення створюється підзображення
    _figure, parent = plt.subplots()

    # Причина, чому ми хочемо побудувати малюнок у зворотному порядку, полягає в тому, що коли ми використовуємо
    # img_to_array, розташування висоти і ширини інвертується. Параметр «origin» задає розташування початкової 
    # точки (0, 0). У цьому випадку - лівий нижній кут
    parent.imshow(img_array[::-1], origin='lower')
    # plt.yticks(visible=False)
    # plt.xticks(visible=False)     
    plt.title(title)

    # Визначте осі вставки на основі батьківських осей і вкажіть значення масштабування (2x, 3x і т.д.)
    # Ми також вказуємо місце для зображення зі зміненим масштабом, у цьому випадку - верхній лівий кут
    inset = zoomed_inset_axes(parent, 2, loc='upper left')
    inset.imshow(img_array[::-1], origin='lower')

    x1, x2, y1, y2 = z1, z2, z3, z4
    inset.set_xlim(x1, x2)  #  Вкажіть координати по осі X для масштабування зображення
    inset.set_ylim(y1, y2)  #  Вкажіть координати по осі Y для масштабування зображення
    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Намалюйте додаткові лінії, що вказують на місце розташування збільшеного зображення
    # «loc1» і «loc2» - кути осей вставки, «fc» і «ec» - кольори ліній
    mark_inset(parent, inset, loc1=1, loc2=3, fc='none', ec='blue')
    plt.savefig(f'images\\{prefix}.jpg')  # Збереження графіка
    plt.close()


def get_zoom_area_indices(image_width, image_height, grid_index, zoom_factor=0.5):
    """
    Функція для обчислення координат для масштабування певної області зображення на основі сітки 5x5.

    Параметри:
    image_width (int): Ширина зображення.
    image_height (int): Висота зображення.
    grid_index (int): Індекс області в сітці 5x5 (від 1 до 25).
    zoom_factor (float): Фактор масштабування (від 0 до 1), де 1 означає повну область.

    Повертає:
    tuple: Координати x1, x2, y1, y2 для області масштабування.
    """
    index_mapping = {
         1: 21,  2: 22,  3: 23,  4: 24,  5: 25,
         6: 16,  7: 17,  8: 18,  9: 19, 10: 20,
        11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
        16:  6, 17:  7, 18:  8, 19:  9, 20: 10,
        21:  1, 22:  2, 23:  3, 24:  4, 25:  5
    }

    mapped_index = index_mapping.get(grid_index, 13)  # За замовчуванням використовуємо центральну область

    row = (mapped_index - 1) // 5
    col = (mapped_index - 1) % 5

    region_width = image_width // 5
    region_height = image_height // 5

    x_center = col * region_width + region_width // 2
    y_center = row * region_height + region_height // 2

    zoom_width = int(region_width * zoom_factor)
    zoom_height = int(region_height * zoom_factor)

    x1 = max(0, x_center - zoom_width // 2)
    x2 = min(image_width, x_center + zoom_width // 2)
    y1 = max(0, y_center - zoom_height // 2)
    y2 = min(image_height, y_center + zoom_height // 2)

    return x1, x2, y1, y2


def send_zoom_options(bot, message):
    """
    Надсилає повідомлення з інлайн-кнопками для вибору області збільшення.
    """
    
    markup = telebot.types.InlineKeyboardMarkup(row_width=5)
    buttons = [telebot.types.InlineKeyboardButton(text=str(i), callback_data=f'zoom_{i}') for i in range(1, 26)]
    buttons.append(telebot.types.InlineKeyboardButton(text='Випадково', callback_data='zoom_custom'))
    markup.add(*buttons)

    bot.send_message(chat_id=message.chat.id, text='Виберіть область, яку треба наблизити', reply_markup=markup)


def help_comparison(bot, model, button_message_id, message):
    # delete 'Скасувати' button after image is in chat
    bot.delete_message(chat_id=message.chat.id, message_id=button_message_id)
    bot.user_data[message.chat.id] = {'client_message': message}
    send_zoom_options(bot, message)


def comparison(bot, model, zoom_area_index, message):
    """
    Візуальне порівняння зображення до та після покращення
    """

    change_working_directory_to_current_file()
    try:
        HR_image = download_image(bot, message)

        # Обробка зображення
        upscale_factor = 3
        LR_image = HR_image.resize((HR_image.size[0] // upscale_factor,
                                    HR_image.size[1] // upscale_factor),
                                    Image.BICUBIC)

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
        output = output.resize(HR_image.size, Image.Resampling.NEAREST)

        cb = cb.resize(output.size, Image.Resampling.BICUBIC)
        cr = cr.resize(output.size, Image.Resampling.BICUBIC)

        ER_image = Image.merge('YCbCr', (output, cb, cr))
        ER_image = ER_image.convert('RGB')

        LR_image = LR_image.resize(ER_image.size, Image.Resampling.BICUBIC)

        # Візуалізація результатів
        image_width, image_height = LR_image.size
        z1, z2, y1, y2 = get_zoom_area_indices(image_width, image_height, zoom_area_index, zoom_factor=1.0)

        plot_results(LR_image, 'LR_image', 'Low Resolution', z1, z2, y1, y2)
        plot_results(ER_image, 'ER_image', 'Enhanced Resolution', z1, z2, y1, y2)
        plot_results(HR_image, 'HR_image', 'Original Image', z1, z2, y1, y2)

        # Розрахунок точності
        LR_arr = img_to_array(LR_image)
        HR_arr = img_to_array(HR_image)
        ER_arr = img_to_array(ER_image)

        mse = MeanSquaredError()
        mae = MeanAbsoluteError()

        # PSNR
        bicubic_psnr = tf.image.psnr(LR_arr, HR_arr, max_val=255)
        test_psnr = tf.image.psnr(ER_arr, HR_arr, max_val=255)
        lr_er_psnr = tf.image.psnr(LR_arr, ER_arr, max_val=255)

        # SSIM
        bicubic_ssim = tf.image.ssim(LR_arr, HR_arr, max_val=255)
        test_ssim = tf.image.ssim(ER_arr, HR_arr, max_val=255)

        # MSE
        bicubic_mse = mse(LR_arr, HR_arr)
        test_mse = mse(ER_arr, HR_arr)

        # MAE
        bicubic_mae = mae(LR_arr, HR_arr)
        test_mae = mae(ER_arr, HR_arr)

        LR_HR = f'PSNR між LR та HR зображеннями \\= {bicubic_psnr:.3f}\n'.replace('.', '\\.')
        ER_HR = f'PSNR між ER та HR зображеннями \\= {test_psnr:.3f}\n'.replace('.', '\\.')
        difference = f'> *Різниця PSNR \\= {(test_psnr - bicubic_psnr):.3f}*\n\n'.replace('.', '\\.').replace('-', '\\- ')
        caption = LR_HR + ER_HR + difference
        # LR_ER = f'\n||PSNR між LR та ER зображеннями \\= {lr_er_psnr:.3f}||'.replace('.', '\\.')
        # caption = LR_HR + ER_HR + difference + LR_ER

        caption += f'SSIM між LR та HR зображеннями \\= {bicubic_ssim:.3f}\n'.replace('.', '\\.')
        caption += f'SSIM між ER та HR зображеннями \\= {test_ssim:.3f}\n'.replace('.', '\\.')
        caption += f'> *Різниця SSIM \\= {(test_ssim - bicubic_ssim):.3f}*\n\n'.replace('.', '\\.').replace('-', '\\- ')

        caption += f'MSE між LR та HR зображеннями \\= {bicubic_mse:.3f}\n'.replace('.', '\\.')
        caption += f'MSE між ER та HR зображеннями \\= {test_mse:.3f}\n'.replace('.', '\\.')
        caption += f'> *Різниця MSE \\= {(test_mse - bicubic_mse):.3f}*\n\n'.replace('.', '\\.').replace('-', '\\- ')

        caption += f'MAE між LR та HR зображеннями \\= {bicubic_mae:.3f}\n'.replace('.', '\\.')
        caption += f'MAE між ER та HR зображеннями \\= {test_mae:.3f}\n'.replace('.', '\\.')
        caption += f'> *Різниця MAE \\= {(test_mae - bicubic_mae):.3f}*\n'.replace('.', '\\.').replace('-', '\\- ')

        # Відправлення зображень
        with open('images\\LR_image.jpg', 'rb') as LR_image, \
                open('images\\ER_image.jpg', 'rb') as ER_image, \
                open('images\\HR_image.jpg', 'rb') as HR_image:
            media = [telebot.types.InputMediaPhoto(LR_image, caption=caption, parse_mode='MarkdownV2'),
                        telebot.types.InputMediaPhoto(ER_image),
                        telebot.types.InputMediaPhoto(HR_image)]
            bot.send_media_group(message.chat.id, media, reply_to_message_id=message.message_id)
    except Exception as e:
        msg = 'На жаль, сталася помилка \nСпробуйте викликати команду /comparison ще раз'
        bot.send_message(message.chat.id, msg)
