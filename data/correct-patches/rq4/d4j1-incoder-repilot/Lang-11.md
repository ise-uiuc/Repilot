# Repilot Patch

```
        } else {
            if (end <= start) {
                throw new IllegalArgumentException("end (" + end + ") must be greater than start (" + start + ")");
            }
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

@@ -241,6 +241,10 @@

                     end = 'z' + 1;
                     start = ' ';                
                 }
+            }
+        } else {
+            if (end <= start) {
+                throw new IllegalArgumentException("end (" + end + ") must be greater than start (" + start + ")");
             }
         }
```

# Note

