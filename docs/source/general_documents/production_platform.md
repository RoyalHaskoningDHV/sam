# Production platform

![alt text](images/technical_overview.png)

## Kafka

We use apache kafka as our main 'message bus' and to provide scalability and consistency. [Read all about it](https://kafka.apache.org/).

### Test setup

One topic called 'sensordata' to send some data, this will be read by a python script, and the message will be repeated on the 'prediction' topic. Right now there is a test setup configured with the security as listed in the next paragraph, and the generated files are on the NAS in the SAM folder as well.

### Python

Both python-kafka and confluent-kafka seem to work and are easy enough to use. You can find examples for both [here](kafka_python.html).

### Security setup

The installation of the whole kafka-cluster is beyond this document, but a few notes on security are included here as it directly affects the consumers/producers.

Kafka supports a few options to make things more secure, [try to read this](https://kafka.apache.org/documentation/#security).

Two ways that I got to work:

- One of the SASL ways, kerberos, using our FreeIPA server and generating a keytab file
- Using certficates (called SSL in the docs), both for client and server authentication

In the final setup we could use both, but most likely using just SSL is enough. This works as follows, these commands are also in [paste 11](https://dev.ynformed.nl/P11):

- We generate our own 'certificate authority': CA (I used openssl tool)
- We generate a public/private key pair for the server (I used java keytool), and stored them in a jks file
- We generate a certicate signing request (keytool)
- We sign the certicate with our CA certificate (openssl), make sure the DNS name is included!
- We load the signed certicate in the jks file, and load the CA public cert

All servers in the cluster should have their own public/private key pair, signed by the same CA. For kafka we would like to end up with a 'keystore' and a 'truststore', both .jks files.

Next we repeat the process of generating a private/public key pair for the client and sign it with the same CA. For the client, at least in python and librdkafka (C client) we want to end up with 'normal' files like 'key', 'crt' and a 'ca' file.

To debug if everything worked we can use the console producer and consumer with the following call:

```
$KAFKA_HOME/bin/kafka-console-producer.sh --topic=sensordata --broker-list sam01.ynformed.nl:9092 --producer.config=/ssl/client-ssl.properties
```

and

```
$KAFKA_HOME/bin/kafka-console-consumer.sh --topic=predictions --bootstrap-server sam01.ynformed.nl:9092 --consumer.config=/ssl/client-ssl.properties
```

This does require the `.jks` files.

