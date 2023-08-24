# Repilot Patch

```
classes[i] = array[i] == null ? null : array[i].getClass();
```

# Developer Patch

```
            classes[i] = array[i] == null ? null : array[i].getClass();
```

# Context

```
--- bug/Lang-33/src/main/java/org/apache/commons/lang3/ClassUtils.java

+++ fix/Lang-33/src/main/java/org/apache/commons/lang3/ClassUtils.java

@@ -907,7 +907,7 @@

         }
         Class<?>[] classes = new Class[array.length];
         for (int i = 0; i < array.length; i++) {
-            classes[i] = array[i].getClass();
+classes[i] = array[i] == null ? null : array[i].getClass();
         }
         return classes;
     }
```

# Note

