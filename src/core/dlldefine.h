#pragma once

#ifndef SUPERSTITCH_EXPORTS
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined(LIBPANOAPI_EXPORTS)
#   define SUPERSTITCH_EXPORTS __declspec(dllexport)
# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(LIBPANOAPI_EXPORTS) || defined(__APPLE__))
#   define SUPERSTITCH_EXPORTS __attribute__ ((visibility ("default")))
# endif
#endif

#ifndef SUPERSTITCH_EXPORTS
# define SUPERSTITCH_EXPORTS
#endif


#ifndef LIBPANO_DEPRECATED
#  if defined(__GNUC__)
#    define LIBPANO_DEPRECATED __attribute__ ((deprecated))
#  elif defined(_MSC_VER)
#    define LIBPANO_DEPRECATED __declspec(deprecated)
#  else
#    define LIBPANO_DEPRECATED
#  endif
#endif

