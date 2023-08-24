# Repilot Patch

```

            this.finish();
```

# Developer Patch

```
            this.finish();
```

# Context

```
--- bug/Compress-1/src/main/java/org/apache/commons/compress/archivers/cpio/CpioArchiveOutputStream.java

+++ fix/Compress-1/src/main/java/org/apache/commons/compress/archivers/cpio/CpioArchiveOutputStream.java

@@ -343,6 +343,8 @@

      */
     public void close() throws IOException {
         if (!this.closed) {
+
+            this.finish();
             super.close();
             this.closed = true;
         }
```

# Note

