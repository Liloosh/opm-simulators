/*
  Copyright 2026 OPM Project Authors

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef OPM_CALLSTACK_LOG_HPP
#define OPM_CALLSTACK_LOG_HPP

#include <iostream>
#include <memory>

#ifndef DUNE_ENABLE_BICGSTAB_CALLSTACK_LOG_DEFINED
#define DUNE_ENABLE_BICGSTAB_CALLSTACK_LOG_DEFINED
namespace Dune {

inline bool& enableBiCGStabCallStackLog() {
    static bool enable = false;
    return enable;
}

inline std::shared_ptr<std::ostream>& getCallStackLogStream() {
    static std::shared_ptr<std::ostream> stream(&std::cout, [](std::ostream*){});
    return stream;
}

} // namespace Dune
#endif

#endif // OPM_CALLSTACK_LOG_HPP
