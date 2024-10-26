Интеллектуальный помощник для создания учебных материалов
**Содержание**
1.
2.
3.
к4
к4

4

**Проблема**
Современный мир с его стремительным ритмом и информационным перегрузом создает трудности в организации материалов и ресурсов при подготовке к важным событиям, таким как переговоры, собеседования и экзамены.

По результатам запущенного нами опроса, мы выявили:
**1.** Из 46 человек 69% тяжело самостоятельно анализировать материалы 
**2.** 21% опрошенных затруднительно анализировать найденную информацию 
**3**. 95% опрошенных считают, что выделение основной информации будет полезно.

**Актуальность**
Эксперимент, проведённый среди студентов учащихся в СПбГУ по анализированию текста. Опрос проведён психологом среди 17 студентов СПбГУ в ноябре 2022 года. Эксперимент представлял собой простой анализ текста на понимание басен Эзопа.

Исходя из результатов теста, можно понять, что современные ученики плохо понимают смысл текста самостоятельно. У них могут возникнуть трудности с подготовкой к важным занятиям или экзаменам из-за неправильного восприятия и большого количества информации.
 из Журнала ВШЭ “Психология”


**Цель**
Создать прототип приложения, которое анализирует методическую информацию (записи лекций и переговоров, конспекты и др.) и генерирует полезные для пользователя материалы.

**Задачи**
**1.** Проанализировать аналоги 
**2.** Создать удобный пользовательский интерфейс 
**3.** Разработать прототип сайта

**Анализ области**

|                |otter.ai    |avtor24.ai              |SaluteSpeech от сбера        |quillionz                     |
|----------------|----------|---------------|--------------|----------|
|Возможности| Из аудио автоматически делает заметки в режиме реального времени          | Загрузка текста и фотографий. Выводит суммаризированный текст           |Распознает только мп3-формат |Автоматически генерирует вопросы для понимания текста, используя ИИ и машинное обучение
|Точность          |90%           |50%          |93% |97% |
|Скорость          |2+ сек|1-2 мин|0.45 - 13.8 сек |5-6 сек |
|Модель ML|модель OpenFlamingo, разработанной DeepMind, и обучена на датасете MultI-Modal In-Context Instruction Tuning (MIMIC-IT)|Текстовые модели GPT‑3.5, GPT‑4|GigaAM, GigaAM - Emo,GigaAM - CTC |GPT‑3.5 |
|+| Расширенные возможности поиска. Универсальная поддержка платформ|Прямое общение авторов с заказчиками. Широкий выбор заданий|Мгновенное озвучивание текста благодаря мощным серверам.  Круглосуточная поддержка и качественная документация|Настраиваемые шаблоны вопросов(для точности содержания) |
|-| Ограниченный бесплатный план| При оплате за работу могут взимать большую комиссию(16-25%)|Платные услуги дорогие для постоянного использования. В редких случаях возможны задержки или сбои в работе сервиса | Ограничение кол-ва символов  |


**Описание предлагаемого решения**

Мы планируем разработать сайт для мобильных и десктопных устройств. В сервисе можно загрузить аудио или видеоматериал (лекции и т.п.) с устройства и в качестве результата получить карточку с основными моментами в виде суммаризированного текстового конспекта.
В качестве дополнительной помощи при обучении сервис, исходя из темы, будет выдавать ссылки на полезные материалы.


**План реализации**

**1.** Выбор технологий
Выбор моделей ML для перевода мультимедиа в текст, суммаризации и тд.
**2.** Проектирование системы
Создание архитектуры сервиса
**3.** Разработка пользовательского интерфейса
Создание макета в фигме.

**Описание отдельных шагов плана**

**1.** Выбор технологий

Для перевода из аудио в текст воспользуемся Whisper AI, это бесплатная нейросеть для создания субтитр и извлечения текста. Для получения текста из видео применим Video Llava. Все выбранные модели отличаются точностью и скоростью работы
Qwen будем использовать для создания краткого конспекта из полученного текста
Для предоставления ссылок пользователю мы подключимся к Perplexity AI,

**2.** Проектирование системы

Архитектура


**3.** Разработка пользовательского интерфейса

Исходя из результатов нашего опроса, большая часть (67,4%) пользуется пк и нотбуками, но при это 32,6% опрошенных удобнее пользоваться телефоном. Следовательно, нужно создать адаптивный под мобильные и десктопные устройства сервис. Разработка дизайна обладает NUI-дизайном, принципом разработки является Mobile First.


При использовании сайта, пользователи должны испытывать комфорт не только от удобного интерфейса, но и от его цветовой палитры. Исходя из статьей различных ученых мы выявили, что для работоспособности человека самыми оптимистическими цветами являются желтый и фиолетовый, которые стимулируют повышению мотивации, креативности, оптимизма и вдохновения.

**Макет интерфейса**

Наш сайт адаптируется под мобильные и десктопные устройства

https://www.figma.com/design/OLOIQbpmNoNC9PEr1IdmsQ/%D1%81%D0%B8%D1%80%D0%B8%D1%83%D1%81.%D0%B8%D0%B8?node-id=0-1&t=JxATM0Rb6LkZ3e3F-1

Наш сайт адаптируется под мобильные и десктопные устройства
