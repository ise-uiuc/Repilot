# Repilot Patch

```
        } else if (end <= start) {
            throw new IllegalArgumentException("Requested range end is less than range start");
```

# Developer Patch

```
        } else {
            if (end <= start) {
                throw new IllegalArgumentException("Parameter end (" + end + ") must be greater than start (" + start + ")");
            }
```

# Context

```
--- bug/Lang-11/src/main/java/org/apache/commons/lang3/RandomStringUtils.java

+++ fix/Lang-11/src/main/java/org/apache/commons/lang3/RandomStringUtils.java

@@ -242,6 +242,8 @@

                     start = ' ';                
                 }
             }
+        } else if (end <= start) {
+            throw new IllegalArgumentException("Requested range end is less than range start");
         }
 
         char[] buffer = new char[count];
```

# Note

