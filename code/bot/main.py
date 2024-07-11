import random
from functools import partial

import telebot
from keras.models import load_model

from config import TOKEN
from control_examples import show_control_examples
from zooming import help_comparison, comparison
from image_enhancement import upgrade_image
from image_degradement import degrade_image
from tools import add_cancel_button, get_messages, change_working_directory_to_current_file


bot = telebot.TeleBot(TOKEN)
bot.set_my_commands([
    telebot.types.BotCommand('/comparison', 'Візуальне порівняння роботи моделі'),
    telebot.types.BotCommand('/control_examples', 'Контрольні приклади'),
    telebot.types.BotCommand('/upgrade', 'Покращити якість фотографії'),
    telebot.types.BotCommand('/degrade', 'Погіршити якість фотографії'),
    telebot.types.BotCommand('/help', 'Допомога (документація)'),
    telebot.types.BotCommand('/theory', 'Теоретичні відомості'),
    telebot.types.BotCommand('/start', 'Привітання'),
])
bot.user_data = {}


change_working_directory_to_current_file()
model = load_model('..\\models\\RDB_ReLU\\RDB_ReLU_40.h5')              #  Diff PSNR = 1.73
# model = load_model('..\\models\\RDB_ReLU\\RDB_ReLU_10.h5')              #  Diff PSNR = 1.33

# model = load_model('..\\models\\RDB_Tanh\\RDB_Tanh_100.h5')             #  Diff PSNR = -0.09
# model = load_model('..\\models\\RDB_Sigmoid\\RDB_Sigmoid_40.h5')        #  Diff PSNR = -0.26


@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, 'Вітаю! Це частина дипломної роботи студента КПІ, КМ-01, Романецького Микити')


@bot.message_handler(commands=['help'])
def handle_help(message):
    help_message = get_messages('messages.txt')['help_message']
    bot.reply_to(message, help_message)


@bot.message_handler(commands=['theory'])
def handle_theory(message):
    theory_message = get_messages('messages.txt')['theory_message']
    bot.reply_to(message, theory_message)
    url = 'https://www.researchgate.net/profile/Hojatollah-Yeganeh-2/publication/283461887/figure/fig5/AS:670025388150786@1536757909078/Comparison-of-SSIM-and-MSE-performances-for-Einstein-image-altered-with-different.png'
    caption = f'Ця картинка запозичена за таким посиланням: {url}'
    with open('images\\theory\\theory_1.png', 'rb') as photo:
        bot.send_photo(message.chat.id, photo, caption=caption)


@bot.message_handler(commands=['comparison'])
def handle_comparison(message):
    msg = bot.reply_to(message, 'Надішліть зображення, яке я маю покращити \nБажано без стиснення (файлом)')
    button = add_cancel_button(bot, message, 'cancel_comparison')
    bot.register_next_step_handler(msg, partial(help_comparison, bot, model, button.message_id))


@bot.message_handler(commands=['control_examples'])
def handle_control_examples(message):
    show_control_examples(bot, model, message)


@bot.message_handler(commands=['upgrade'])
def handle_upgrade(message):
    msg = bot.reply_to(message, 'Надішліть зображення, яке я маю покращити \nБажано без стиснення (файлом)')
    button = add_cancel_button(bot, message, 'cancel_upgrade')
    bot.register_next_step_handler(msg, partial(upgrade_image, bot, model, button.message_id))


@bot.message_handler(commands=['degrade'])
def handle_degrade(message):
    msg = bot.reply_to(message, 'Надішліть зображення, яке я маю погіршити \nБажано без стиснення (файлом)')
    button = add_cancel_button(bot, message, 'cancel_degrade')
    bot.register_next_step_handler(msg, partial(degrade_image, bot, button.message_id))


@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    if call.data in ['cancel_comparison', 'cancel_upgrade', 'cancel_degrade']:
        bot.answer_callback_query(call.id, 'Ви скасували надсилання зображення')
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        bot.clear_step_handler_by_chat_id(call.message.chat.id)
    if call.data.startswith('zoom_'):
        zoom_area = int(call.data.split('_')[1]) if call.data.split('_')[1].isdigit() else random.randint(1, 25)
        client_message = bot.user_data[call.message.chat.id]['client_message']
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        comparison(bot, model, zoom_area, client_message)


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, 'Я розумію лише команди')


if __name__ == '__main__':
    bot.polling(none_stop=True)
