import pika
import json
from config import RABBITMQ_URL

def send_message(queue_name, message):
    """
    Send a message to the specified RabbitMQ queue.

    :param queue_name: The name of the RabbitMQ queue.
    :param message: The message to send (will be converted to JSON).
    """
    connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    channel = connection.channel()

    channel.queue_declare(queue=queue_name, durable=True)

    message_json = json.dumps(message)
    channel.basic_publish(
        exchange='',
        routing_key=queue_name,
        body=message_json,
        properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
    )

    connection.close()

def receive_messages(queue_name, callback):
    """
    Receive messages from the specified RabbitMQ queue.

    :param queue_name: The name of the RabbitMQ queue.
    :param callback: The callback function to handle received messages.
    """
    connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    channel = connection.channel()

    channel.queue_declare(queue=queue_name, durable=True)

    def wrapped_callback(ch, method, properties, body):
        message = json.loads(body)
        callback(message)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=wrapped_callback)

    channel.start_consuming()
