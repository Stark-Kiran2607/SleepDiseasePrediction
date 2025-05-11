drop database if exists sleep_db;
create database sleep_db;
use sleep_db;


create table users(
    id INT PRIMARY KEY AUTO_INCREMENT, 
    name VARCHAR(50), 
    email VARCHAR(50), 
    password VARCHAR(50)
    );