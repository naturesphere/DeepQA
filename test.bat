:: 看时间
::echo %date%

:: 建立新数据库，即pkl文件
::python .\main.py --corpus lightweight --datasetTag lwc2 --createDataset --playDataset

:: 训练模型
python .\main.py --corpus lightweight --datasetTag lwc2
