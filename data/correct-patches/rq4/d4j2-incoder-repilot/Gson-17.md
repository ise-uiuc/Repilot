# Repilot Patch

```
    if (in.peek() == JsonToken.NULL) {
      in.nextNull();
      return null;
```

# Developer Patch

```
    if (in.peek() == JsonToken.NULL) {
      in.nextNull();
      return null;
```

# Context

```
--- bug/Gson-17/gson/src/main/java/com/google/gson/DefaultDateTypeAdapter.java

+++ fix/Gson-17/gson/src/main/java/com/google/gson/DefaultDateTypeAdapter.java

@@ -96,8 +96,9 @@

 
   @Override
   public Date read(JsonReader in) throws IOException {
-    if (in.peek() != JsonToken.STRING) {
-      throw new JsonParseException("The date should be a string value");
+    if (in.peek() == JsonToken.NULL) {
+      in.nextNull();
+      return null;
     }
     Date date = deserializeToDate(in.nextString());
     if (dateType == Date.class) {
```

# Note

