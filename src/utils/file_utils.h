#pragma once

#ifndef _BASIC_FILE_UTAILS_H_
#define _BASIC_FILE_UTAILS_H_


#pragma warning(disable:4996)

#include <string>
#include <vector>

#include "dlldefine.h"

#include <opencv2/opencv.hpp>


namespace libpano
{

	bool SUPERSTITCH_EXPORTS is_file(const std::string& filename);
	bool SUPERSTITCH_EXPORTS delete_file(const std::string& filename);

	bool SUPERSTITCH_EXPORTS is_directory(const std::string& filename);
	bool SUPERSTITCH_EXPORTS delete_directory(const std::string& path);
	bool SUPERSTITCH_EXPORTS create_directory(const std::string& path);

	void SUPERSTITCH_EXPORTS get_directory_entries(const std::string& dir, std::vector<std::string>& entries);


	std::string SUPERSTITCH_EXPORTS get_current_working_directory();

	bool	SUPERSTITCH_EXPORTS	set_current_working_directory(const std::string& path);

	std::string SUPERSTITCH_EXPORTS get_home_directory(void);

	bool SUPERSTITCH_EXPORTS rename_file(const std::string& old_name, const std::string& new_name);

	time_t	SUPERSTITCH_EXPORTS get_time_stamp(const std::string& file_or_dir);
	std::string SUPERSTITCH_EXPORTS get_time_string(const std::string& file_or_dir);

	std::string SUPERSTITCH_EXPORTS convert_to_lower_case(const std::string& str);
	std::string SUPERSTITCH_EXPORTS convert_to_upper_case(const std::string& str);

	std::string SUPERSTITCH_EXPORTS dir_name(const std::string& file_name);

	std::string extension(const std::string& file_name);

	std::string SUPERSTITCH_EXPORTS extension_in_lower_case(const std::string& filename);

	std::string SUPERSTITCH_EXPORTS simple_name(const std::string& file_name);

	std::string SUPERSTITCH_EXPORTS base_name(const std::string& file_name);

	std::string SUPERSTITCH_EXPORTS name_less_extension(const std::string& file_name);

	std::string SUPERSTITCH_EXPORTS name_less_all_extensions(const std::string& file_name);

	std::string SUPERSTITCH_EXPORTS replace_extension(std::string const& file_name, std::string const& ext);

	std::string SUPERSTITCH_EXPORTS get_path_root(const std::string& path);

	bool SUPERSTITCH_EXPORTS is_absolute_path(const std::string& path);

	std::string SUPERSTITCH_EXPORTS get_relative_path(const std::string& from, const std::string& to);

	std::string SUPERSTITCH_EXPORTS get_absolute_path(const std::string& path);

	std::string SUPERSTITCH_EXPORTS convert_to_windows_style(const std::string& path);

	std::string SUPERSTITCH_EXPORTS convert_to_unix_style(const std::string& path);

	char SUPERSTITCH_EXPORTS get_native_path_separator();


	bool SUPERSTITCH_EXPORTS is_native_style(const std::string& path);


	std::string SUPERSTITCH_EXPORTS convert_to_native_style(const std::string& path);

	void SUPERSTITCH_EXPORTS get_directory_entries(const std::string& dir, std::vector<std::string>& entries, bool recursive);
	void SUPERSTITCH_EXPORTS get_files(const std::string& dir, std::vector<std::string>& files, bool recursive = false);
	void SUPERSTITCH_EXPORTS get_subdirectories(const std::string& dir, std::vector<std::string>& subs, bool recursive = false);

	bool SUPERSTITCH_EXPORTS copy_file(const std::string& original, const std::string& copy);
	bool SUPERSTITCH_EXPORTS file_contains_string(const std::string& file_name, const std::string& x);

	void SUPERSTITCH_EXPORTS read_file_to_string(const std::string& filename, std::string& data);
	void SUPERSTITCH_EXPORTS write_string_to_file(const std::string& data, const std::string& filename);
	void SUPERSTITCH_EXPORTS write_string_to_file(const char* data, int len, const std::string& filename);

	void SUPERSTITCH_EXPORTS get_filenames_with_extension(
		const std::string& dir, std::vector<std::string>& filenames,
		const std::string& extension = "*.jpg");

	void SUPERSTITCH_EXPORTS get_filenames_with_absolute_path(
		const std::string& dir, std::vector<std::string>& filenames,
		const std::string& extension = "*.jpg");


}//namespace libpano




#endif
