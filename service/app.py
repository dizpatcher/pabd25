from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    
    try:
        data = request.get_json()
        print(data)
        required_fields = ['number_of_rooms', 'area', 'flat_floor', 'total_floors']

        # Проверка наличия всех полей
        for field in required_fields:
            if field not in data:
                return {'status': 'error', 'message': f'Missing field: {field}'}

        # Преобразование и валидация значений
        numbers = []
        for field in required_fields:
            try:
                num = float(data[field])
                if num < 0:
                    return {'status': 'error', 'message': f'{field} должно быть положительным'}
                
                if field == 'number_of_rooms' and num > 4:
                    return {'status': 'error', 'message': 'Количество комнат не может быть больше четырёх'}
                
                if field == 'flat_floor' and num > float(data['total_floors']):
                    return {'status': 'error', 'message': 'Этаж квартиры не может быть больше общего количества этажей'}
                
                if field == 'area' and num > 500:
                    return {'status': 'error', 'message': 'Площадь квартиры не может превышать 1000 м²'}
                
                numbers.append(num)
            except ValueError:
                return {'status': 'error', 'message': f'{field} должно быть числом'}
        
        # Если всё ок — возвращаем полученные данные
        return {'status': 'success', 'data': numbers}

    except Exception as e:
        return {'status': 'error', 'message': f'Ошибка сервера: {str(e)}'}

if __name__ == '__main__':
    app.run(debug=True)
