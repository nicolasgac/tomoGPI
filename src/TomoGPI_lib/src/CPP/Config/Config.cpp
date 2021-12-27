/*
 * Config.cpp
 *
 *      Author: gac
 */

#include "Config.hpp"

/* Config definition */
Config::Config() : configFileName(string()){}
Config::Config(string configDirectoryName, string configFileName) : configDirectoryName(configDirectoryName), configFileName(configFileName){}
Config::~Config(){	configDirectoryName.clear();configFileName.clear();}

string Config::getConfigFileName() const
{
	return configFileName;
}

string Config::getConfigDirectoryName() const
{
	return configDirectoryName;
}

void Config::setConfigFileName(string configFileName)
{
	this->configFileName = configFileName;
}

void Config::setConfigDirectoryName(string configDirectoryName)
{
	this->configDirectoryName = configDirectoryName;
}

template <>
string Config::getConfigFileField(string field)
{

	ifstream configFile;

	configFile.open((this->configDirectoryName+this->configFileName).c_str(), ios::in);
	string row;
	string temp;
	string extractField;

	if (configFile.is_open())
	{
		while (getline(configFile, row))
		{
			istringstream rowss(row, istringstream::in);
			while(rowss >> temp)
			{
				if (temp==field)
				{
					rowss >> extractField;
					configFile.close();
					return extractField;
				}
			}
		}
		configFile.close();
		cout << "Unable to find field " << field << endl;
		exit(EXIT_FAILURE);
	}
	else
	{
		cout << "Unable to open file " << this->configDirectoryName+this->configFileName << endl;
		exit(EXIT_FAILURE);
	}
}

template <>
int Config::getConfigFileField(string field)
{

	ifstream configFile;

	configFile.open((this->configDirectoryName+this->configFileName).c_str(), ios::in);
	string row;
	string temp;
	string extractField;

	if (configFile.is_open())
	{
		while (getline(configFile, row))
		{
			istringstream rowss(row, istringstream::in);
			while(rowss >> temp)
			{
				if (temp==field)
				{
					rowss >> extractField;
					configFile.close();
					return atoi(extractField.c_str());
				}
			}
		}
		configFile.close();
		cout << "Unable to find field " << field << endl;
		exit(EXIT_FAILURE);
	}
	else
	{
		cout << "Unable to open file " << this->configDirectoryName+this->configFileName << endl;
		exit(EXIT_FAILURE);
	}
}

template <>
float Config::getConfigFileField(string field)
{

	ifstream configFile;

	configFile.open((this->configDirectoryName+this->configFileName).c_str(), ios::in);
	string row;
	string temp;
	string extractField;

	if (configFile.is_open())
	{
		while (getline(configFile, row))
		{
			istringstream rowss(row, istringstream::in);
			while(rowss >> temp)
			{
				if (temp==field)
				{
					rowss >> extractField;
					configFile.close();
					return atof(extractField.c_str());
				}
			}
		}
		configFile.close();
		cout << "Unable to find field " << field << endl;
		exit(EXIT_FAILURE);
	}
	else
	{
		cout << "Unable to open file " << this->configDirectoryName+this->configFileName << endl;
		exit(EXIT_FAILURE);
	}
}

template <>
double Config::getConfigFileField(string field)
{

	ifstream configFile;

	configFile.open((this->configDirectoryName+this->configFileName).c_str(), ios::in);
	string row;
	string temp;
	string extractField;

	if (configFile.is_open())
	{
		while (getline(configFile, row))
		{
			istringstream rowss(row, istringstream::in);
			while(rowss >> temp)
			{
				if (temp==field)
				{
					rowss >> extractField;
					configFile.close();
					return atof(extractField.c_str());
				}
			}
		}
		configFile.close();
		cout << "Unable to find field " << field << endl;
		exit(EXIT_FAILURE);
	}
	else
	{
		cout << "Unable to open file " << this->configDirectoryName+this->configFileName << endl;
		exit(EXIT_FAILURE);
	}
}

void Config::copyConfigFile(string destinationDirectory)
{
	ifstream configFile;
	ofstream copyConfigFile;
	int i;

	configFile.open((this->configDirectoryName+this->configFileName).c_str(), ios::in);
	if (configFile.is_open())
	{
		copyConfigFile.open((destinationDirectory+this->configFileName).c_str(), ios::out | ios::trunc);
		if (copyConfigFile.is_open())
		{
			copyConfigFile << configFile.rdbuf();
			copyConfigFile.close();
			string name = "chmod 774 ";
			i=system((name + (destinationDirectory+this->configFileName).c_str()).c_str());
		}
		else
		{
			cout << "Unable to open file " << destinationDirectory+this->configFileName << endl;
			exit(EXIT_FAILURE);
		}
		configFile.close();
	}
	else
	{
		cout << "Unable to open file " << this->configDirectoryName+this->configFileName << endl;
		exit(EXIT_FAILURE);
	}
}





