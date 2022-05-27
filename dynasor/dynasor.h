
/***************************************************************************************/
/*                                                                                     */
/*                conceptual dynamic tensor library implementation                     */
/*                                                                                     */
/*                                Pooria Yousefi                                       */
/*                             pooriayousefi@aol.com                                   */
/*                    https://www.linkedin.com/in/pooriayousefi/                       */
/*                                     2022                                            */
/*                                                                                     */
/***************************************************************************************/

#pragma once

#include <type_traits>
#include <algorithm>
#include <numeric>
#include <vector>
#include <execution>
#include <random>
#include <fstream>
#include <iostream>
#include <functional>
#include <concepts>

// anonymous space
namespace
{
	// ------------------------------------------------
	//
	//              concepts definitions
	//
	// ------------------------------------------------

	// arithmetic concept
	template<typename T>
	concept arithmetic = std::integral<T> || std::floating_point<T>;

	// integral value iterator concept
	template<typename T>
	concept integral_value_iterator = std::input_or_output_iterator<T> && std::integral<std::iter_value_t<T> >;

	// floating-point value iterator concept
	template<typename T>
	concept floating_point_value_iterator = std::input_or_output_iterator<T> && std::floating_point<std::iter_value_t<T> >;

	// arithmetic value iterator concept
	template<typename T>
	concept arithmetic_value_iterator = integral_value_iterator<T> || floating_point_value_iterator<T>;

	// execution policy concept
	template<typename T>
	concept execution_policy = std::is_execution_policy_v<T>;

	// ------------------------------------------------
	//
	//                 dynamic tensor
	//
	// ------------------------------------------------
	template<arithmetic T>
	class dynasor
	{
	public:
		// type definitions
		using type = dynasor<T>;

		// default constructor
		dynasor() :number_of_elements_in_dimension_{}, data_{}{}

		// constructor
		template<execution_policy ExPo, integral_value_iterator DimIter>
		dynasor(ExPo expo, DimIter dimbeg, DimIter dimend ) 
			:number_of_elements_in_dimension_(std::distance(dimbeg, dimend), 0ULL), data_()
		{
			std::copy(
				expo,
				dimbeg,
				dimend,
				number_of_elements_in_dimension_.begin()
			);
			auto data_size{
				std::reduce(
					expo, 
					number_of_elements_in_dimension_.begin(), 
					number_of_elements_in_dimension_.end(), 
					1ULL, 
					std::multiplies<size_t>()
				)
			};
			data_.resize(data_size, (T)0);
		}

		// constructor
		template<execution_policy ExPo, integral_value_iterator DimIter>
		dynasor(ExPo expo, DimIter dimbeg, DimIter dimend, T initval) 
			:number_of_elements_in_dimension_(std::distance(dimbeg, dimend), 0ULL), data_()
		{
			std::copy(
				expo,
				dimbeg,
				dimend,
				number_of_elements_in_dimension_.begin()
			);
			auto data_size{
				std::reduce(
					expo,
					number_of_elements_in_dimension_.begin(),
					number_of_elements_in_dimension_.end(),
					1ULL,
					std::multiplies<size_t>()
				)
			};
			data_.resize(data_size, initval);
		}

		// constructor
		template<execution_policy ExPo, integral_value_iterator DimIter, arithmetic_value_iterator ValIter>
		dynasor(ExPo expo, DimIter dimbeg, DimIter dimend, ValIter valbeg, ValIter valend) 
			:number_of_elements_in_dimension_(std::distance(dimbeg, dimend), 0ULL), data_(std::distance(valbeg, valend), (T)0)
		{
			std::copy(
				expo,
				dimbeg,
				dimend,
				number_of_elements_in_dimension_.begin()				
			);
			std::copy(
				expo,
				valbeg,
				valend,
				data_.begin()
			);
		}

		// constructor
		template<execution_policy ExPo, integral_value_iterator DimIter, typename F, typename... Args>
		dynasor(ExPo expo, DimIter dimbeg, DimIter dimend, F&& f, Args&&... args) 
			:number_of_elements_in_dimension_(std::distance(dimbeg, dimend), 0ULL), data_()
		{
			if constexpr (std::is_invocable_r_v<T, F, Args...>)
			{
				std::copy(
					expo,
					dimbeg,
					dimend,
					number_of_elements_in_dimension_.begin()
				);
				auto data_size{
					std::reduce(
						expo,
						number_of_elements_in_dimension_.begin(),
						number_of_elements_in_dimension_.end(),
						1ULL,
						std::multiplies<size_t>()
					)
				};
				data_.resize(data_size, (T)0);
				std::generate(
					std::execution::seq,
					data_.begin(),
					data_.end(),
					[&]() { return std::forward<F>(f)(std::forward<Args>(args)...); }
				);
			}			
		}

		// copy constructor
		dynasor(const dynasor&) = default;

		// move constructor
		dynasor(dynasor&&) noexcept = default;

		// destructor
		virtual ~dynasor() = default;

		// overloaded copy assignment operator
		auto operator=(const dynasor&)->dynasor & = default;

		// overloaded move assignment operator
		auto operator=(dynasor&&) noexcept->dynasor & = default;

		// overloaded call operator
		template<integral_value_iterator IdxIter>
		auto operator()(IdxIter idxbeg, IdxIter idxend)->T &
		{
			return data_[index(idxbeg, idxend)];
		}

		// overloaded call operator
		template<integral_value_iterator IdxIter>
		auto operator()(IdxIter idxbeg, IdxIter idxend) const->T
		{
			return data_[index(idxbeg, idxend)];
		}

		// number of tensor dimensions getter method
		auto number_of_dimensions() const->size_t { return number_of_elements_in_dimension_.size(); }

		// data accessor
		auto data()->std::vector<T> & { return data_; }

		// N-Dimensional index space ---> 1-Dimensional index space converter
		template<integral_value_iterator IdxIter>
		auto index(IdxIter idxbeg, IdxIter idxend) const->size_t
		{			
			if (std::distance(idxbeg, idxend) == number_of_dimensions()) {
				auto nd{ number_of_dimensions() };
				auto I{ 0ULL };
				for (auto d{ 0ULL }; d <= nd - 2ULL; ++d) {
					auto nm{ 1ULL };
					for (auto m{ d + 1ULL }; m <= nd - 1ULL; ++m)
						nm *= number_of_elements_in_dimension_[m];
					I += (*std::next(idxbeg, d) * nm);
				}
				I += *std::next(idxbeg, nd - 1ULL);
				return I;
			}
			else
				throw std::runtime_error("ERROR: indexes range size is not equal to number of tensor dimensions.");
		}

		// tensor element (in N-Dimensional index space) data accessor
		template<integral_value_iterator IdxIter>
		auto element(IdxIter idxbeg, IdxIter idxend)->T&
		{
			return data_[index(idxbeg, idxend)];
		}

		// tensor element (in N-Dimensional index space) data getter method
		template<integral_value_iterator IdxIter>
		auto element(IdxIter idxbeg, IdxIter idxend) const->T
		{
			return data_[index(idxbeg, idxend)];
		}

		// zero valued dynamic tensor factory
		template<execution_policy ExPo, integral_value_iterator DimIter>
		static dynasor zeros(ExPo expo, DimIter dimbeg, DimIter dimend) { return dynasor(expo, dimbeg, dimend, (T)0); }
		
		// one valued dynamic tensor factory
		template<execution_policy ExPo, integral_value_iterator DimIter>
		static dynasor ones(ExPo expo, DimIter dimbeg, DimIter dimend) { return dynasor(expo, dimbeg, dimend, (T)1); }
		
		// uniformly random valued dynamic tensor factory
		template<execution_policy ExPo, integral_value_iterator DimIter>
		static dynasor uniform_random(ExPo expo, DimIter dimbeg, DimIter dimend, size_t seed, T param1, T param2)
		{
			auto _{ dynasor<T>::zeros(expo, dimbeg, dimend) };
			std::mt19937_64 rng{};
			rng.seed(seed);
			if constexpr (std::is_integral_v<T>)
			{
				std::uniform_int_distribution<T> rnd(param1, param2);
				std::generate(
					std::execution::seq,
					_.data().begin(),
					_.data().end(),
					[&rng, &rnd]() { return rnd(rng); }
				);
			}
			if constexpr (std::is_floating_point_v<T>)
			{
				std::uniform_real_distribution<T> rnd(param1, param2);
				std::generate(
					std::execution::seq,
					_.data().begin(),
					_.data().end(),
					[&rng, &rnd]() { return rnd(rng); }
				);
			}
			return _;
		}
		
		// normal random valued dynamic tensor factory
		template<execution_policy ExPo, integral_value_iterator DimIter>
		static dynasor normal_random(ExPo expo, DimIter dimbeg, DimIter dimend, size_t seed, T param1, T param2)
		{
			auto _{ dynasor<T>::zeros(expo, dimbeg, dimend) };
			std::mt19937_64 rng{};
			rng.seed(seed);
			std::normal_distribution<T> rnd(param1, param2);
			std::generate(
				std::execution::seq,
				_.data().begin(),
				_.data().end(),
				[&rng, &rnd]() { return rnd(rng); }
			);
			return _;
		}
		
		// Gaussian random valued dynamic tensor factory
		template<execution_policy ExPo, integral_value_iterator DimIter>
		static dynasor gaussian_random(ExPo expo, DimIter dimbeg, DimIter dimend, size_t seed, T param1, T param2)
		{
			return dynasor<T>::normal_random(expo, dimbeg, dimend, seed, param1, param2);
		}

	private:
		std::vector<size_t> number_of_elements_in_dimension_;
		std::vector<T> data_;
	};
}
