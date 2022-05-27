
#include <initializer_list>
#include <set>
#include "dynasor.h"

// entry point
auto main()->int
{
	try
	{
		// dynamic tensor tester
		auto dynasor_test = []()
		{
			std::initializer_list<size_t> idx0{ 2, 3, 1, 5 };
			auto FIRST0{ idx0.begin() };
			auto LAST0{ idx0.end() };
			auto SEQ0{ std::execution::seq };
			dynasor<float> dt0(SEQ0, FIRST0, LAST0);

			std::set<short> idx1{ 3, 2 };
			auto FIRST1{ idx1.begin() };
			auto LAST1{ idx1.end() };
			auto PAR1{ std::execution::par };
			auto dt1{ dynasor<int>::uniform_random(PAR1, FIRST1, LAST1, 4373, -2, 1) };
		};
		dynasor_test();

		return EXIT_SUCCESS;
	}
	catch (const std::exception & xxx)
	{
		std::cerr << xxx.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (...)
	{
		std::cerr << "UNCAUGHT EXCEPTION DETECTED" << std::endl;
		return EXIT_FAILURE;
	}
}
