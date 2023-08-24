# Repilot Patch

```

        // Trim trailing NULs...
```

# Developer Patch

```

```

# Context

```
--- bug/Compress-27/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

+++ fix/Compress-27/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

@@ -130,10 +130,8 @@

             end--;
             trailer = buffer[end - 1];
         }
-        if (start == end) {
-            throw new IllegalArgumentException(
-                    exceptionMessage(buffer, offset, length, start, trailer));
-        }
+
+        // Trim trailing NULs...
 
         for ( ;start < end; start++) {
             final byte currentByte = buffer[start];
```

# Note

