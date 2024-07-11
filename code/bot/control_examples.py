import matplotlib
matplotlib.use('Agg')

import numpy as np
import telebot
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.losses import MeanSquaredError, MeanAbsoluteError

from zooming import get_zoom_area_indices, plot_results
from tools import change_working_directory_to_current_file


def show_control_examples(bot, model, message):
    counter = 1
    content = [['images\\control_examples\\raw\\1_HR_control_example.jpg', 18],
               ['images\\control_examples\\raw\\2_HR_control_example.jpg', 8],
               ['images\\control_examples\\raw\\3_HR_control_example.jpg', 17],
               ['images\\control_examples\\raw\\4_HR_control_example.jpg', 13],
               ['images\\control_examples\\raw\\5_HR_control_example.jpg', 14],
               ['images\\control_examples\\raw\\6_HR_control_example.jpg', 21],
               ['images\\control_examples\\raw\\7_HR_control_example.jpg', 24],
               ]

    try:
        change_working_directory_to_current_file()
        
        for path, zoom_area_index in content:
            HR_image = Image.open(path).convert('RGB')

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

            plot_results(LR_image, f'control_examples\\done\\{counter}_LR_control_example', 
                        'Low Resolution', z1, z2, y1, y2)
            plot_results(ER_image, f'control_examples\\done\\{counter}_ER_control_example', 
                        'Enhanced Resolution', z1, z2, y1, y2)
            plot_results(HR_image, f'control_examples\\done\\{counter}_HR_control_example', 
                        'Original Image', z1, z2, y1, y2)

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
            with open(f'images\\control_examples\\done\\{counter}_LR_control_example.jpg', 'rb') as LR_image, \
                    open(f'images\\control_examples\\done\\{counter}_ER_control_example.jpg', 'rb') as ER_image, \
                    open(f'images\\control_examples\\done\\{counter}_HR_control_example.jpg', 'rb') as HR_image:
                media = [telebot.types.InputMediaPhoto(LR_image, caption=caption, parse_mode='MarkdownV2'),
                            telebot.types.InputMediaPhoto(ER_image),
                            telebot.types.InputMediaPhoto(HR_image)]
                bot.send_media_group(message.chat.id, media, reply_to_message_id=message.message_id)
            counter += 1
    except Exception as e:
        msg = 'На жаль, сталася помилка \nСпробуйте викликати команду /control_examples ще раз'
        bot.send_message(message.chat.id, msg)
